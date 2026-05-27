# 문서 인덱스

이 폴더는 사람이 읽는 문서를 둔다. 기본 언어는 한국어다.

## 문서

- `INSTALL.md`: conda 환경 설치, editable install, 검증 명령어
- `WORKFLOW.md`: 전체 workflow 목차와 현재 진행 상태
- `STATUS_2026-05-18.md`: 2026-05-18 기준 구현/결정/검증/남은 작업 스냅샷
- `STATUS_2026-05-21.md`: DAgger v0, pretrained loading, replay, logger 상태 스냅샷
- `PROJECT_SCOPE.md`: 프로젝트 범위, 제외 항목, 현재 scaffold
- `PIPELINE.md`: online intervention step 흐름과 demo/intervention buffer routing
- `DAGGER_BASELINE.md`: learner rollout + expert relabel 방식의 DAgger baseline
- `PRETRAINED_POLICIES.md`: 현재 사용 가능한 pretrained expert/learner weight와 로드 방법
- `REPLAY_AND_UPDATES.md`: replay schema, n-step backup, UTD, target Q aggregation
- `NETWORKS.md`: 공용 MLP 옵션과 알고리즘별 network 기본값
- `JAX_FLAX_GUIDE.md`: Torch 사용자 관점에서 보는 JAX/Flax, `TrainState`, gradient 흐름
- `LOGGING_AND_METRICS.md`: interval 평균 로깅 방식과 추가 metric TODO
- `REAL_ENV_SMOKE_TESTS.md`: 실제 Robomimic env에서 돌린 smoke test 명령과 결과
- `CODE_REVIEW_2026-05-26.md`: `il/` 전체 코드 리뷰 finding과 2026-05-27 기준 해결/미해결 상태
- `RUN_FAILURE_LOG.md`: 실행 실패와 원인/후속 조치 기록
- `RESIDUAL_POLICY_ANALYSIS.md`: ResFiT residual policy 분석과 우리 레포 통합 설계 메모
- `EXTENSIBILITY_REVIEW_2026-05-21.md`: 현재 hardcoded default/prefix와 확장성 리스크 점검
- `IMPLEMENTATION_PLAN.md`: 구현 순서와 검증 상태
- `CONFIG_SCHEMA_DECISIONS.md`: config schema, DAgger v0, intervention/gate, replay sampling 결정 사항

## 작성 규칙

- 실험 의도와 의사결정 배경을 한국어로 적는다.
- 실행 명령어와 결과 요약은 사람이 빠르게 확인할 수 있게 적는다.
- 큰 그림은 `WORKFLOW.md`에 두고, 세부 내용은 작업 단위별 문서에 둔다.
- Codex/Claude에게 줄 세부 작업 지시는 `docs_agents/`에 영어로 분리한다.

## 최근 추가 config

- `config/smoke_residual_square.yaml`: frozen BCFlow base + residual RLPD learner rollout smoke
