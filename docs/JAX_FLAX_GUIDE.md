# Torch 사용자를 위한 JAX/Flax 코드 읽기 가이드

이 문서는 PyTorch는 익숙하지만 JAX/Flax는 익숙하지 않은 상태에서 `intervention_learning` 코드를 읽기 위한 가이드다. 현재 repo는 qc/qc_base 스타일을 가져왔기 때문에 `ModuleDict`, `TrainState`, `flax.struct.PyTreeNode`를 이해하면 대부분의 학습 코드를 읽을 수 있다.

## 1. 큰 그림

PyTorch에서는 보통 module이 parameter를 내부에 들고 있고, forward에서 그 parameter를 쓰면 autograd가 추적한다.

```python
model = Agent()
optimizer = torch.optim.Adam(model.parameters())

q = model.critic(obs, act)
loss = ((q - target) ** 2).mean()
loss.backward()
optimizer.step()
```

JAX/Flax에서는 네트워크를 function처럼 보고, parameter를 함수 입력으로 명시적으로 넣는다.

```python
def loss_fn(params):
    q = model.apply({"params": params}, obs, act)
    loss = ((q - target) ** 2).mean()
    return loss

grads = jax.grad(loss_fn)(params)
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

핵심 차이는 다음과 같다.

```text
PyTorch: module이 parameter를 소유하고 autograd가 추적한다.
JAX: loss_fn의 입력으로 들어간 params에 대해 gradient를 계산한다.
```

## 2. PyTree

JAX는 array 하나뿐 아니라 dict/list/tuple/dataclass처럼 중첩된 구조 전체를 parameter로 다룬다. 이런 중첩 구조를 PyTree라고 부른다.

이 repo의 RLPD agent에서 `self.network.params`는 대략 다음 구조다.

```python
{
    "modules_critic": ...,
    "modules_target_critic": ...,
    "modules_actor": ...,
    "modules_alpha": ...,
}
```

BC flow agent에서는 대략 다음 구조가 된다.

```python
{
    "modules_actor_bc_flow": ...,
}
```

`jax.grad(loss_fn)(self.params)`를 호출하면 이 전체 tree와 같은 구조의 gradient tree가 나온다.

## 3. `nn.Module`

`flax.linen.nn.Module`은 PyTorch의 `nn.Module`과 이름은 비슷하지만 사용 감각이 다르다. Flax module은 네트워크 구조를 정의하고, 실제 weight는 `params` PyTree로 따로 관리한다.

이 repo의 공용 wrapper는 `il/utils/flax_utils.py`의 `ModuleDict`다.

```python
class ModuleDict(nn.Module):
    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            ...
        return self.modules[name](*args, **kwargs)
```

`ModuleDict`는 actor, critic, target critic 같은 여러 Flax module을 하나로 묶는다. 아래처럼 `select`로 어느 module을 forward할지 고른다.

```python
self.network.select("actor")(obs)
self.network.select("critic")(obs, act)
self.network.select("target_critic")(next_obs, next_act)
```

중요한 점은 `select`는 module 선택만 한다는 것이다. gradient를 흘릴지 말지는 `params=grad_params`를 넘기는지로 결정된다.

## 4. `flax.struct.PyTreeNode`

`flax.struct.PyTreeNode`는 JAX 변환 함수인 `jit`, `grad`, `scan` 안에서 안전하게 들고 다닐 수 있는 immutable dataclass에 가깝다.

RLPD agent는 `il/algo/rl/rlpd.py`에서 다음처럼 정의된다.

```python
class ACRLPDAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
```

`rng`, `network`는 JAX가 PyTree로 다룬다. `config`는 일반 Python object라서 `nonpytree_field()`로 빼둔다.

PyTreeNode는 직접 값을 바꾸기보다 `replace`로 새 객체를 반환한다.

```python
return self.replace(network=new_network, rng=new_rng), info
```

Torch에서는 `optimizer.step()`이 model을 in-place로 바꾸는 감각이 강하다. JAX에서는 update 후 새 agent/state를 반환하는 functional style에 가깝다.

## 5. `TrainState`

이 repo의 핵심 wrapper는 `il/utils/flax_utils.py`의 `TrainState`다.

```python
class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    grad_clip_norm: Any = nonpytree_field(default=None)
```

역할은 다음과 같다.

```text
model_def: Flax nn.Module 구조
apply_fn: model_def.apply 함수
params: 실제 weight PyTree
tx: optax optimizer
opt_state: Adam moment 같은 optimizer state
step: optimizer step counter
grad_clip_norm: optional global grad clipping
```

Torch로 비유하면 `nn.Module + optimizer + optimizer_state + step`을 한 객체에 담은 것이다. 단, parameter는 module 안이 아니라 `params` 필드에 있다.

## 6. Forward와 gradient 흐름

`TrainState.__call__`은 Flax apply를 감싼다.

```python
def __call__(self, *args, params=None, method=None, **kwargs):
    if params is None:
        params = self.params
    variables = {"params": params}
    return self.apply_fn(variables, *args, method=method_name, **kwargs)
```

따라서 아래 두 코드는 의미가 다르다.

```python
# loss_fn의 입력인 grad_params를 사용한다.
# 이 module parameter에 gradient가 생긴다.
self.network.select("critic")(obs, act, params=grad_params)

# TrainState가 저장한 self.params를 사용한다.
# loss_fn의 입력인 grad_params를 직접 쓰지 않으므로 이 module parameter gradient는 생기지 않는다.
self.network.select("critic")(obs, act)
```

`grad_params`는 bool flag가 아니다. `jax.grad(loss_fn)(self.params)`에서 JAX가 미분 대상으로 받은 parameter tree 그 자체다.

## 7. RLPD update 흐름

RLPD update는 `il/algo/rl/rlpd.py`에서 다음 흐름이다.

```python
def loss_fn(grad_params):
    return self.total_loss(batch, grad_params, rng=rng)

new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
self.target_update(new_network, "critic")
```

`TrainState.apply_loss_fn` 내부에서는 전체 params에 대해 gradient를 구한다.

```python
grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
    grads, _ = optax.clip_by_global_norm(self.grad_clip_norm).update(grads, None, self.params)

return self.apply_gradients(grads=grads, update_scale=update_scale), info
```

그 다음 optimizer를 적용한다.

```python
updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
new_params = optax.apply_updates(self.params, updates)
```

Torch로 치면 다음과 대응된다.

```python
loss = loss_fn(model.parameters())
loss.backward()
optimizer.step()
target_critic = tau * critic + (1 - tau) * target_critic
```

## 8. RLPD critic loss

critic target을 만들 때 actor와 target critic에는 `params=grad_params`를 넘기지 않는다.

```python
next_dist = self.network.select("actor")(next_obs)
next_actions = next_dist.sample(seed=sample_rng)
next_qs = self.network.select("target_critic")(next_obs, next_actions)
```

이 부분은 Torch의 `with torch.no_grad()`에 가깝다.

현재 critic을 학습할 때만 `params=grad_params`를 넘긴다.

```python
q = self.network.select("critic")(obs, actions, params=grad_params)
critic_loss = ((q - target_q) ** 2).mean()
```

그래서 critic loss는 critic parameter만 업데이트한다.

## 9. RLPD actor loss

actor는 `params=grad_params`로 호출한다.

```python
dist = self.network.select("actor")(obs, params=grad_params)
actions = dist.sample(seed=rng)
log_probs = dist.log_prob(actions)
```

critic은 `params=grad_params` 없이 호출한다.

```python
qs = self.network.select("critic")(obs, actions)
q = jnp.mean(qs, axis=0)
actor_loss = (log_probs * alpha - q).mean()
```

결과는 다음과 같다.

```text
critic weight에는 gradient가 흐르지 않는다.
하지만 Q(obs, actions)는 actions의 함수이므로 dQ/da는 actor로 흐른다.
actor weight에는 gradient가 흐른다.
```

Torch로 비유하면 다음과 비슷하다.

```python
actions = actor(obs)

for p in critic.parameters():
    p.requires_grad_(False)

q = critic(obs, actions)
actor_loss = -q.mean()
actor_loss.backward()
```

## 10. BC Flow update 흐름

BC flow는 `il/algo/bc/flow.py`에 있다. 구조는 RLPD보다 단순하다.

```python
def loss_fn(grad_params):
    return agent.total_loss(batch, grad_params, rng=rng)

new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
```

BC flow는 actor BC flow module만 가지고 있으므로, 보통 `grad_params` 전체가 곧 `modules_actor_bc_flow` 학습 대상이다.

DAgger에서는 rollout은 learner policy로 하고, expert action을 저장한 뒤 BC target으로 쓴다. 그래서 학습 관점에서는 RL critic update가 아니라 supervised BC update다.

## 11. optimizer 분리 여부

현재 `intervention_learning`의 RLPD는 optimizer를 하나만 만든다.

```python
network_tx = optax.adam(learning_rate=config["lr"])
```

따라서 actor/critic optimizer state가 분리되어 있지는 않다. 대신 `params=grad_params`를 어디에 넘기는지로 실제 gradient 흐름을 조절한다.

나중에 discriminator, curiosity, auxiliary value 등을 붙일 때는 두 가지 선택지가 있다.

```text
1. 하나의 TrainState에 넣고 optax.multi_transform으로 optimizer group을 분리한다.
2. 별도 TrainState/Agent로 분리하고 update order를 명확히 한다.
```

teacher/student 관계, target network, discriminator처럼 update 순서가 중요한 경우는 joint loss보다 별도 update가 안전하다.

## 12. `jit`, `grad`, `scan`

자주 나오는 JAX 변환은 다음 세 개다.

```text
jax.grad(f): f의 입력에 대한 gradient 함수를 만든다.
jax.jit(f): f를 XLA로 컴파일한다.
jax.lax.scan(f): Python loop 대신 컴파일 가능한 loop를 만든다.
```

RLPD의 `batch_update`는 UTD처럼 여러 update를 묶을 때 `lax.scan`을 쓴다.

```python
agent, infos = jax.lax.scan(self._update, self, batch)
return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
```

`scan`은 Torch의 Python for-loop와 비슷한 목적이지만, JAX가 통째로 컴파일할 수 있게 만든 loop라고 보면 된다.

## 13. 코드 읽을 때 체크리스트

JAX/Flax loss를 읽을 때는 아래 순서로 확인하면 된다.

```text
1. 어떤 agent인가? ACRLPDAgent, BCFlowAgent 등.
2. self.network가 어떤 ModuleDict를 들고 있는가?
3. self.network.select("...")로 어떤 module을 호출하는가?
4. 그 호출에 params=grad_params가 들어가는가?
5. 들어가면 해당 module parameter에 gradient가 흐른다.
6. 안 들어가면 해당 module parameter는 frozen evaluator처럼 쓰인다.
7. 그래도 입력 tensor가 grad_params에서 나온 값이면 입력 방향 gradient는 흐를 수 있다.
8. optimizer가 단일 Adam인지 multi_transform인지 확인한다.
9. target network가 optimizer step 뒤 Polyak update되는지 확인한다.
```

이 기준으로 보면 JAX/Flax 코드도 Torch의 `requires_grad`, `detach`, `no_grad`, `optimizer.step`과 대응시켜 읽을 수 있다.
