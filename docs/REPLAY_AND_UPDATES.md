# Replay와 Update 설계

## Transition Schema

필수 필드:

- `observations`
- `actions`
- `learner_actions`
- `expert_actions`
- `rewards`
- `terminals`
- `masks`
- `next_observations`
- `controller_ids`
- `episode_ids`
- `episode_steps`
- `gating_reasons`
- `gating_scores`
- `learner_action_log_probs`
- `expert_action_log_probs`
- `interventions`

`actions`는 항상 env에 실제 실행된 action이다.

`episode_ids`는 transition이 속한 episode 번호이고, `episode_steps`는 해당 episode 안에서의
0-based step index다. image observation을 next frame으로 재구성할 때도 이 두 값을 사용한다.

`controller_ids`:

- `0`: learner
- `1`: expert

`gating_reasons`:

- `0`: none
- `1`: random gate

## Terminal과 Mask

`terminals`는 timeout/truncation을 포함해서 episode가 끝났는지를 나타낸다.

`masks`는 true termination에 대해서만 0이 된다. timeout/truncation에서는 나중에 bootstrap을
할 수 있게 1을 유지하는 방향이다.

## Image Replay Storage

image observation은 `observations`와 `next_observations`에 중복 저장하지 않는다. ReplayBuffer는
image leaf를 내부적으로 `image_observations[camera_name]`에 current frame으로 한 번만 저장한다.

저장 구조 예시:

```python
buffer.data["observations"]["state"]         # low-dim state
buffer.image_data["agentview"]               # current image frame
buffer.image_data["sideview"]
```

sample할 때 반환 batch는 기존 API를 유지한다.

```python
batch["observations"]["state"]
batch["observations"]["agentview"]           # image at transition i
batch["next_observations"]["state"]
batch["next_observations"]["agentview"]      # image from transition i + 1
batch["image_next_valid"]                    # i+1 image가 같은 episode의 다음 step이면 1
```

`next_observations`의 image는 `episode_ids`가 같고 `episode_steps`가 정확히 `+1`인 다음 transition에서
가져온다. 아직 다음 transition이 buffer에 없거나 episode boundary를 넘으면 image는 zero frame으로
채우고 `image_next_valid=0`으로 표시한다.

## N-step Backup

QC의 핵심 학습 방식처럼 replay buffer에서 `sample_sequence(batch_size, n, gamma)`를 제공한다.
이 함수는 연속 transition을 뽑고 discounted n-step reward를 계산해서 agent update에 넘긴다.

목표 target 형태:

```text
target = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1}
       + gamma^n * mask_chain * Q_target(s_{t+n}, a_{t+n})
```

v0에서는 `horizon_length=1`로 시작하지만, buffer API는 처음부터 n-step을 받을 수 있게 둔다.
action chunking은 나중에 다루더라도 n-step backup 자체는 유지한다.

## Episode Boundary 처리

v0에서는 `qc_base`와 동일하게 episode boundary를 n-step window 중간에 포함한 sample은
학습에서 쓰지 않는다. 구현상 sample은 뽑히지만 `valid[..., -1] = 0`이 되어 critic loss에
기여하지 않는다.

예를 들어 `n=5`인데 두 번째 transition에서 episode가 끝나면, 그 sample로 짧은
`n=2` target을 만들지 않고 버린다.

나중에 value 학습이나 sample efficiency가 중요해지면 다음 대안을 추가한다.

```text
boundary_mode = "drop"
  qc_base 방식. 중간 boundary가 있으면 sample을 loss에서 제외한다.

boundary_mode = "truncate"
  boundary 지점에서 짧은 n-step target을 만든다.
  true termination이면 bootstrap하지 않고,
  timeout/truncation이면 boundary next_observation에서 bootstrap한다.
```

예를 들어 `n=5`로 뽑았는데 `n=2` 지점에서 timeout이면:

```text
target = r_0 + gamma r_1 + gamma^2 Q(s_2)
```

## Update-to-data Ratio

UTD는 한 env step마다 몇 번 gradient update를 할지 정하는 값이다.

```text
utd_ratio = 1
  env step 1개당 update 1번

utd_ratio = 4
  env step 1개당 update 4번
```

구현상 replay에서 `batch_size * utd_ratio`만큼 뽑은 뒤,
`(utd_ratio, batch_size, ...)` 형태로 reshape해서 `agent.batch_update()`에 넘긴다.

## Mixed Buffer Sampling

물리적 buffer는 역할별로 나눈다.

- `online`: learner가 online rollout 중 모은 모든 transition
- `intervention`: expert intervention이 발생한 episode의 first-intervention 이후 suffix
- `demo`: autonomous success trajectory, offline expert demo, scripted expert demo 등 clean expert label source

학습 batch를 만들 때는 `MixedReplaySampler`가 지정 비율대로 여러 buffer에서 샘플을 뽑아 합친다.

예시:

```python
buffers = ReplayBufferCollection(
    online=online_buffer,
    intervention=intervention_buffer,
    demo=demo_buffer,
)

sampler = MixedReplaySampler(
    buffers,
    MixedSamplingSpec({"online": 0.5, "intervention": 0.25, "demo": 0.25}),
)

batch = sampler.sample_sequence(batch_size=256, sequence_length=1, discount=0.99)
```

`TrainingConfig.sampling_fractions`를 쓰면 `sample_rl_update_batch()`에서도 같은 방식으로 섞을 수 있다.

```python
config = TrainingConfig(
    batch_size=256,
    utd_ratio=1,
    horizon_length=1,
    discount=0.99,
    sampling_fractions={"online": 0.5, "intervention": 0.25, "demo": 0.25},
)
batch = sample_rl_update_batch(buffers, config)
```

비율은 꼭 합이 1일 필요는 없다. 내부에서 normalize한 뒤 integer count로 변환한다.
나머지 sample은 fractional remainder가 큰 buffer에 배정해서 총 batch size를 정확히 맞춘다.

예시 조건:

- online + intervention에서 반반: `{"online": 0.5, "intervention": 0.5}`
- intervention + demo에서 반반: `{"intervention": 0.5, "demo": 0.5}`
- 세 buffer에서 1/3씩: `{"online": 1, "intervention": 1, "demo": 1}`

## Initial Replay Prefill

online/intervention/demo buffer는 optional하게 시작 시점에 기존 replay dataset으로 채울 수 있다.
현재 지원하는 포맷은 `ReplayBuffer.save_npz()`로 저장한 schema-compatible `.npz`다.

예시:

```yaml
replay:
  online_size: 500000
  intervention_size: 500000
  demo_size: 500000
  prefill:
    demo:
      path: /path/to/demo_replay.npz
      format: npz
      max_transitions: 100000  # optional
```

`prefill`은 `online`, `intervention`, `demo` 중 원하는 buffer에 걸 수 있다. 물리 buffer는 계속
분리되어 있고, prefill은 해당 buffer의 초기 `size/pointer`만 채운다.
간단한 경우 `demo: /path/to/demo_replay.npz`처럼 path string만 넣어도 된다.

image replay도 같은 방식으로 로드된다. `.npz` 안의 `image_observations/<camera>` key는
`ReplayBuffer.image_data[camera]`로 복원되고, sample 시 `i+1` frame으로 next image를 재구성한다.

## Demo Episode Insert Mode

demo buffer는 완전한 expert dataset이라고 가정하지 않는다. online 중 더 짧은 성공 trajectory가
나오면 기존 demo pool의 긴 성공 trajectory를 밀어낼 수 있다.

지원 모드:

- `none`: online 중 성공한 episode도 demo buffer에는 넣지 않는다. prefill/offline demo를 고정하고 싶을 때 쓴다.
- `append`: 기존처럼 성공 episode를 transition 단위로 뒤에 추가한다.
- `replace_longest_if_better`: 여유 공간 여부와 무관하게, 현재 저장된 가장 긴 episode보다 새 성공
  episode가 짧을 때 그 episode를 제거하고 새 episode를 넣는다. 비교할 기존 episode가 없으면 append한다.

예시:

```python
route_episode_to_buffers(
    episode,
    demo_buffer=demo_buffer,
    intervention_buffer=intervention_buffer,
    include_failed_interventions=False,
    demo_insert_mode="replace_longest_if_better",
)
```

이 동작은 `episode_ids`, `episode_steps`가 올바르게 저장되어 있어야 한다. 특히 prefill된 demo와
online rollout의 `episode_ids`가 충돌하지 않게 관리해야 한다.

구현상 `replace_longest_if_better`를 쓰는 순간 `ReplayBuffer`가 episode index를 만든다.
index는 `episode_id -> length/indices`와 longest-first 정렬 리스트를 캐시한다. 그래서 매번 전체
buffer를 훑어 worst episode를 찾는 방식이 아니라, replay mutation 이후 필요한 시점에만 index를
재구성하고 이후에는 정렬된 worst list에서 고른다. 현재 worst metric은 `length`다.

## TD Target Q Aggregation

critic target을 만들 때 target critic ensemble을 하나의 Q 값으로 줄인다.

```text
target_q_agg = "mean"
  q_tar1, q_tar2의 평균을 TD target에 사용한다.

target_q_agg = "min"
  q_tar1, q_tar2 중 minimum을 TD target에 사용한다. 기본값.
```

`num_qs`는 critic ensemble 크기라서 조정 가능하지만, TD target 계산은 항상 앞의 2개만
사용한다. 따라서 `num_qs >= 2`가 필요하다.

## Frame Stack

frame stack은 과거 observation을 현재 observation과 이어 붙이는 기능이다.

v0 기본값은 `frame_stack=1`이다. 즉 현재 observation만 사용한다. 나중에 image 기반
실험이나 partial observability 문제가 생기면 `frame_stack > 1` 구현을 확장한다.
