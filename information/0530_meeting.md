# 0530 Meeting

[Timeline](https://www.notion.so/366f83920e1e80f59158d3a5e6beb1fe?pvs=21)

지난 미팅(0520) 이후 진행 사항 요약. 큰 방향은 그대로 — T2I → **Multi-Image to Video**.
이번 사이클은 **Inference strategy 고도화 + design-choice ablation 본격화 + 정량 평가체계 구축**에 집중.

* 핵심 진척
  1. Image-first inference의 train/inference 불일치를 줄이는 **Bridge forward process** 설계·구현.
  2. Object/style 유지를 위한 **Persistent anchor conditioning** 도입.
  3. **경량화**: temporal cross-attn 제거로 1.14B → **0.59B** (약 −48%), capacity가 병목이 아님을 확인.
  4. **VAE flicker 진단** → 현재 시스템에서 병목 아님(VAE_OK).
  5. **정량 평가 harness** 구축(지난 Metric TODO 대응): CLIP/anchor-fidelity/flow-warp + paired bootstrap 의사결정.
  6. 위 design choice들을 **one-axis-at-a-time ablation**으로 구성하여 학습 중.

지난 미팅 TODO 대응:

| 지난 TODO | 이번 진행 |
|---|---|
| 1.1B 파라미터 경량화 | cross-attn 제거(E1) + mid/up placement 제한 (§3) |
| Inference strategy → object style 유지 | Bridge process(§1) + Persistent anchor(§2) |
| temporal-consistency aux loss(대안) | flow-warp 기반 진단 후 필요 시 도입으로 보류(§4) |
| Frame condition 실험 | sinusoidal `add_to_text` 고정, cross-attn 잉여성 확인(§3) |
| Metric (FVD/IS/CLIPSIM) | 평가 harness 구축(§5) — CLIP/flow-warp proxy 우선, FVD는 옵션 |

---

## 1. Inference Strategy 고도화 — Bridge Forward Process

지난 미팅의 image-first(anchor에서 시작해 frame-wise 확장)에서 **train/inference 불일치**가 핵심 문제로 드러남.

* Training: true video latent를 forward process로 corrupt → denoise
* Inference: anchor를 일부 denoise한 **intermediate latent**에서 시작
  → 둘이 서로 다른 manifold 위에 존재 가능.

**해결 — Smooth-SNR bridge.** noising source를 anchor와 정답 latent 사이에서 SNR에 따라 부드럽게 섞음.

$$
g(t)=\tfrac12\!\left(1+\cos\pi\cdot\text{pos}(t)\right)\ \ (\text{log-SNR }1\!\to\!5),\qquad
\text{source}_t = g(t)\,\mathbb E(a) + (1-g(t))\,z^*
$$

$$
z_t = \sqrt{\bar\alpha_t}\,\text{source}_t + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad
\text{target}=\epsilon\ \text{that reconstructs } z^*
$$

* **Boundary loss** (SNR=5): pred-$x_0$를 경계에서 재노이즈해 $z^*$의 저주파와 일치시켜 anchor↔video 전환을 매끄럽게.
* **pred\_x0\_renoise switch**: inference 전환점에서 anchor를 그대로 복제하는 대신 **base UNet으로 pred-$x_0$를 추정 후 scheduler 노이즈 레벨로 재노이즈** → training forward와 분포 정합.

```
[기존] repeat_add_noise:  z_switch = repeat(z_img) + small_noise
[개선] pred_x0_renoise:    eps = base_sdxl(z_img, t);  x0 = z_img - σ·eps
                           z_switch = √ᾱ·x0 + √(1-ᾱ)·η   (frame-wise η)
```

---

## 2. Persistent Anchor Conditioning (object/style 유지)

지난 미팅 TODO인 "inference 시 object style 유지". 기존 설계는 anchor가 **noising source로만** 들어가서, denoise chain 내내 stage-1 이미지를 참조할 메커니즘이 없었음.

* **개선**: temporal attention에 **gated, 공간 정렬된 anchor K/V token** 추가.
  각 spatial site의 frame들이 anchor의 같은 site를 참조 → identity/layout 유지.

```python
# VideoBasicTransformerBlock 내부
temporal = temporal_self_attn(x)                 # 기존 motion 경로
anchor_kv = proj(resize(anchor_latent))          # anchor를 블록 해상도로 투영
out = temporal + gate * anchor_attn(q, anchor_kv)  # gate=0 zero-init → 초기엔 identity
```

* gate=0 zero-init이라 학습된 checkpoint에서 **continuation fine-tune** 시 출력 불변에서 출발 → gate가 열리며 anchor 정보 주입.
* Training anchor = clean $z^*_1$, Inference anchor = stage-1 pred-$x_0$ (학습/추론 정합).

---

## 3. 경량화 — Attention Adapter

지난 미팅 TODO. "그냥 줄이기"가 아니라 **capacity가 어디에 필요한가**로 재정의.

* per-block temporal adapter 분해(C=hidden width 기준):
  temporal self-attn ≈ 4C², temporal **cross-attn ≈ 4C²**, temporal FFN ≈ 0.69C²(이미 low-rank).
* `token_embedding_mode=add_to_text`에서는 frame 정보가 이미 text 경로 + temporal self-attn으로 들어가므로 **sinusoidal token에 대한 temporal cross-attn은 잉여**.

**E1 (cross-attn 제거):**

```
Default trainable total:   1,140,072,518   (1.140B)
E1 (no temporal cross-attn):  594,235,718   (0.594B)   ← −48%
```

* 추가: adapter를 **mid/up block에만** 주입(placement 제한)으로 추가 절감 예정.
* 참고: 1.1B는 비교군(AnimateDiff motion module ~0.4B, SimDA 24M) 대비 **과도** → capacity가 아니라 data/평가가 병목.

---

## 4. VAE Temporal Flicker 진단

"temporal-consistency aux loss 도입" 전에, frozen·frame-independent VAE decoder가 실제 flicker 병목인지 먼저 측정.

* **지표**: RAFT optical-flow로 인접 프레임을 warp 후, **occlusion mask** 적용한 VGG/Charbonnier 잔차의 "초과분"
  $$\text{excess}=E(\text{decode}_{t+1},\text{warp}(\text{decode}_t))-E(\text{gt}_{t+1},\text{warp}(\text{gt}_t))$$
* **결과 (300 clips, 512², 8f)**: Charbonnier excess **−5.3% (평균)**, p90 +7%, 90% 클립이 음수 → **VAE_OK** (VAE는 flicker를 추가하기보다 오히려 약하게 smoothing).
* 결론: 현재 시스템에서 VAE decoder는 binding bottleneck 아님 → decoder 학습/교체는 보류. (고주파 텍스처 콘텐츠에서만 tail 존재 → 추후 재점검)

---

## 5. 정량 평가 Harness (지난 Metric TODO)

train loss / 단일 프롬프트 eyeballing으로는 **inference에서만 발현되는 축**(switch / bridge / boundary)을 판정 불가 → 정량 평가 체계 구축.

* **지표** (clip별)
  * `clip_t`: frame↔text CLIP cosine (텍스트 정합)
  * `clip_anchor` / `vgg_anchor` / `pix_anchor_low`: frame0 ↔ stage-1 anchor 충실도
  * `warp_vgg`: RAFT flow-warp 후 occlusion-masked VGG 잔차 (temporal flicker)
  * `motion_mag` / `motion_cov`: 움직임 크기·정지영상 guardrail
* **의사결정 규칙(paired)**: baseline 대비 prompt별 delta → 20% trimmed mean → 10k bootstrap 95% CI → baseline seed-repeat로 추정한 `σ_null` 대비 **|delta| > 2σ_null AND CI가 0 제외**일 때만 "유의미".
* **벤치마크**: synthetic 24 + **held-out OpenVid caption 72 = 96 프롬프트** (camera-motion 버킷 균형).
* 모든 모델 결과를 **하나의 wandb run**에 묶어 side-by-side 비교(영상 그리드 + 지표 테이블).
* FVD/IS(I3D/C3D)는 옵션으로 남김 — 내부 ranking엔 CLIP+flow-warp가 비용대비 우수.

---

## 6. Design-Choice Ablation (현재 학습 중)

baseline 대비 **한 축씩만** 다르게 하여 효과를 귀속.

| run | bridge | boundary | switch(infer) | cross-attn | anchor | 비고 |
|---|---|---|---|---|---|---|
| baseline | smooth | 0.05 | pred_x0_renoise | ✓ | — | 기준 (15k 완료) |
| **E1** | smooth | 0.05 | pred_x0_renoise | ✗ | — | 경량화 0.59B |
| smooth_snr_boundary | smooth | 0.05 | repeat | ✓ | — | switch 효과 |
| smooth_snr | smooth | 0.0 | repeat | ✓ | — | boundary 효과 |
| snr_renoise | hard | 0.0 | pred_x0_renoise | ✓ | — | smooth vs hard gate |
| **E2** anchor | smooth | 0.05 | pred_x0_renoise | ✗ | ✓ | persistent anchor |
| **E3** calib | smooth | 0.05 | pred_x0_renoise | ✗ | ±  | train/infer manifold 보정 |
| **E4** rollout | smooth | 0.05 | pred_x0_renoise | ✓ | — | bridge source = rollout pred-x0 |

* 비교 구도: `E1 vs baseline`(cross-attn 제거), `E2 vs E1`(anchor), `E3 vs E2`(calibrator), `E4 vs baseline`(rollout source).
* 현재 E1 외 3개 ablation 동시 학습 중(H200 4-GPU/실험, 15k, ETA ~19h). E2/E3/E4는 코드 준비 완료.

---

## 7. 방법론 검증 (2nd-opinion agent와의 feasibility 점검)

* 현재 budget(130k clips, eff-batch 64, 15k step, 1.1→0.6B)은 비교군 대비 **screening엔 충분, 미세 ranking엔 부족** → 정량 평가(§5)가 결정적.
* train/loss로는 switch/bridge/boundary를 못 가림 → multi-prompt paired 평가 필수(반영 완료).
* capacity가 아니라 **data/eval-limited** → 경량화 방향(§3)과 일관.

---

## Current Status

* baseline(0520) 재현 완료, 15k.
* design ablation 4종 학습 중 (디스크 정리 후 재시작, ckpt 한도 설정으로 안정화).
* Image-first/anchor/calibrator/rollout 코드 및 평가 harness, 96-prompt 벤치마크 구축 완료.
* multi-server 이식성: dataset 경로·python 환경을 `.env`로 일원화.

## TODOs (다음)

1. **평가 실행 → design 결정**: open_clip 설치 후 평가 harness로 6+개 체크포인트 paired 비교 → cross-attn 제거 / bridge / boundary / switch 확정.
2. **E2/E3/E4 학습** 후 anchor·calibrator·rollout 효과 측정 (object/style 유지 정량화).
3. **경량화 2단계**: mid/up placement + self-attn weight sharing.
4. **진짜 held-out 학습**: dataloader에 해시 파티션 제외 → 평가셋을 unseen으로.
5. **baseline reproduction + FVD/IS** 숫자 비교(필요 시 I3D/C3D feature extractor 도입).
