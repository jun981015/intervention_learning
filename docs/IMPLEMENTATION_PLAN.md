# 구현 계획과 검증 상태

## 완료된 Scaffold

- Project scaffold와 replay schema를 만들었다.
- Random gate와 policy interface를 만들었다.
- N-step `ReplayBuffer.sample_sequence()` smoke test를 만들었다.
- `qc_base` 기준 최소 RLPD/SAC agent adapter를 가져왔다.
- BC Flow actor adapter를 가져왔다.
- BC MLP actor adapter를 가져왔다.
- `il/` 아래 class/function docstring을 채웠다.
- simulator 없이 intervention routing smoke test를 추가했다.
- online/demo/intervention buffer를 비율대로 섞는 mixed replay sampler를 추가했다.
- `RLPDPolicy.from_checkpoint()` generic loader를 추가했다.
- `BCFlowPolicy.from_checkpoint()` generic loader를 추가했다.
- RLPD/BC Flow checkpoint save-load smoke test를 추가했다.
- DAgger baseline용 learner rollout + expert relabel helper를 추가했다.
- BC Flow에 `target_action_key` 옵션을 추가했다.

## 현재 검증 명령어

```bash
conda run -n il python -m compileall -q il scripts
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n il python scripts/smoke_test.py
```

현재 smoke 결과:

```text
gate/replay smoke ok
intervention routing smoke ok
mixed replay sampling smoke ok
dagger baseline smoke ok
rlpd smoke ok
rlpd policy checkpoint smoke ok
bc mlp smoke ok
bc flow smoke ok
bc flow policy checkpoint smoke ok
```

## 다음 구현 순서

1. Robomimic Square environment construction을 추가한다.
2. Fresh learner 또는 restored learner + restored expert + random gate로 100 step online rollout smoke를 통과시킨다.
3. replay 저장 파일에 learner/expert/action/gate metadata가 모두 들어가는지 검증한다.
4. No-intervention RLPD baseline을 재현한다.
5. Random intervention baseline을 돌린다.
6. demo/intervention buffer에서 BC loss를 learner update에 추가한다.
7. 그 다음에 smarter gating, human UI, action chunking을 붙인다.

## 아직 안 된 것

- 실제 Robomimic env에서 online rollout이 도는지 확인
- replay buffer save/load round-trip test
- online training loop CLI
- demo/intervention BC loss가 learner update에 들어가는 경로

## Git 운영

- 이 repo는 `/home/junhyeong/repos/intervention_learning`에서 독립 git으로 관리한다.
- remote는 아직 연결하지 않는다.
- GitHub SSH 설정이 끝나면 그때 remote를 추가하고 push한다.
- weight, replay, video, log, wandb 산출물은 절대 커밋하지 않는다.
- 큰 변경은 작은 커밋으로 쪼갠다.
