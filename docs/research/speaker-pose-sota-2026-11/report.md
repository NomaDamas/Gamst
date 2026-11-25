# 세 과제 모두 '논문 SOTA'와 '쓸 수 있는 SOTA'가 갈린다

2026년 11월 기준, 세 과제 모두 논문 1위 모델과 GAMST가 실제로 쓸 모델이 다르다. **화자 검증**의 정확도 1위는 w2v-BERT 2.0 + LoRA-MFA다(VoxCeleb1-O/E/H EER 0.12/0.27/0.55%). 하지만 코드에 라이선스가 없다. 그래서 쓸 수 있는 SOTA는 MIT 라이선스의 **ReDimNet2-B6**이다. 경량 스트리밍용은 sherpa-onnx로 돌리는 **CAM++**이다. **한국어 ASR**은 **Qwen3-ASR-1.7B**(Apache-2.0, FLEURS-ko CER 2.57)와 **Qwen3-ForcedAligner-0.6B**(한국어 단어 시각 오차 37.2 ms)가 정확도와 실용성 모두 1위다. 저지연 스트리밍은 **nemotron-3.5-asr-streaming-0.6b**가 맡는다. **자세 추정**의 정확도 1위는 Meta **Sapiens2**다. 다만 라이선스가 "biometric processing"과 감시 용도를 금지한다. 그래서 기본값은 Apache-2.0인 **RTMW/RTMPose(rtmlib)**와 **MediaPipe**가 맞다. **원샷 반복 동작 계수**는 바로 쓸 수 있는 5k-star급 오픈소스가 없다. 자세 키포인트, 예시와의 유사도, 피크/DTW 계수를 GAMST가 직접 조립해야 한다. 한국어 화자 검증, 한국어 키워드 바이어싱, 스트리머 제스처 계수에는 공개 벤치마크 수치가 하나도 없다. 그래서 어떤 선택이든 사내 한국어 데이터로 잰 지표 게이트를 통과해야 한다. 이 보고서의 수치는 모두 출처에서 인용한 값이다. 이번 조사에서 직접 잰 수치는 없다. GitHub star와 라이선스는 2026-11-25에 `gh api`로 읽었다.

## 화자 검증: 1위는 라이선스가 없고, 실용 1위는 12.3M짜리 ReDimNet2-B6

VoxCeleb 기준 최고 수치는 대형 SSL 백본에서 나온다. **w2v-BERT 2.0 + LoRA_Adapter_MFA**(580M + 6.2M)는 VoxCeleb2와 VoxBlink2로 학습하고 LMFT와 보정을 거쳐 **Vox1-O/E/H EER 0.12/0.27/0.55%**를 낸다. CN-Celeb에서는 4.67%다 ([arXiv 2510.04213](https://arxiv.org/html/2510.04213v2)). 그러나 저자 코드 저장소(ZXHY-82/w2v-BERT-2.0_SV)는 61 star이고 GitHub가 라이선스를 감지하지 못했다. 저작권 전부 보유로 봐야 한다. 같은 백본을 쓴 WeSpeaker 레시피는 **0.250/0.398/0.838**까지 재현했다. 다만 ONNX 내보내기는 없다 ([WeSpeaker voxceleb/v2](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)).

실용 1위는 **ReDimNet2-B6**(PalabraAI, Interspeech 2026)다. 파라미터 12.3M, 13.05 GMACs다. VoxCeleb2만으로 학습해도 **0.29/0.52/0.99**이고, VoxBlink2와 VoxCeleb2로 학습하면 **0.23/0.35/0.67**이다 ([PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)). WeSpeaker가 공식 체크포인트로 0.276/0.518/1.003을 재현했다 ([WeSpeaker](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2)). 저장소는 125 star로 기준(5k)에 못 미친다. 하지만 1.5k star 공식 툴킷인 WeSpeaker가 재현했다는 점이 신뢰를 보탠다. GAMST에 가장 중요한 수치는 CN-Celeb이다. CN-Celeb은 예능과 라이브 방송을 포함한 다장르 세트라 스트리머 음성에 가장 가까운 공개 대용 세트다. 여기서 B6은 CN-Celeb2를 학습에 넣으면 **7.49% → 3.59%**로 떨어진다. 백본 차이보다 도메인 데이터가 훨씬 크게 작용한다는 뜻이다. 문제는 이 vb2·cnc2 가중치가 데이터셋 라이선스를 물려받을 수 있다는 점이다. WeSpeaker는 "사전학습 모델은 해당 데이터셋 라이선스를 따른다"고 명시한다 ([WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)). VoxBlink2가 비상업(CC BY-NC-SA)이라는 기억은 이번에 확인하지 못했다.

라이선스가 가장 깨끗한 중형 대안은 **WeSpeaker ResNet293-LM**이다(28.62M, AS-Norm+QMF로 0.425/0.641/1.146). VoxCeleb 가중치는 CC BY 4.0이고 sherpa-onnx에 109 MB ONNX가 있다. 경량 스트리밍 기본값은 **CAM++**다. 7.18M, 1.72 GFLOPs, 단일 스레드 CPU RTF 0.013으로 ECAPA(0.033)보다 두 배 넘게 빠르다 ([CAM++ arXiv 2303.00332](https://arxiv.org/abs/2303.00332)). sherpa-onnx(16.2k star, Apache-2.0)가 중영 혼합 `zh_en_16k-common_advanced` 26 MB ONNX를 배포한다 ([sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)). 다만 3D-Speaker 저장소는 2025-12-08 이후 push가 없다.

| 역할 | 모델 | 크기 | EER % (Vox1-O / E / H) | CN-Celeb | 라이선스 · star | 플래그 |
|---|---|---|---|---|---|---|
| 정확도 SOTA | w2v-BERT 2.0 + LoRA-MFA | 580M+6.2M | 0.12 / 0.27 / 0.55 | 4.67 | 코드 라이선스 없음 · 61 | **라이선스 없음**, 연구용만 |
| 쓸 수 있는 SOTA | ReDimNet2-B6 (vox2) | 12.3M | 0.29 / 0.52 / 0.99 | – | MIT · 125 (WeSpeaker 재현) | 5k 미만 |
| 도메인 강화 후보 | ReDimNet2-B6 (vb2+vox2+cnc2) | 12.3M | 0.26 / 0.41 / 0.79 | 3.59 | MIT 코드, 가중치는 데이터 라이선스 | **VoxBlink2/CN-Celeb 비상업 가능성 [미확인]** |
| 다국어 후보 | WeSpeaker SimAM-ResNet100 (VB2) | 50.2M | 0.202 / 0.421 / 0.795 | – | ModelScope ONNX · 1.5k | **VoxBlink2 라이선스 [미확인]** |
| 상업 안전 중형 | WeSpeaker ResNet293-LM | 28.62M | 0.425 / 0.641 / 1.146 | – | CC BY 4.0 · 1.5k | 영어 데이터만 |
| 경량 스트리밍 | CAM++ (3D-Speaker, sherpa-onnx) | 7.18M | 0.65 (3D-Speaker 표) | 6.78 | Apache-2.0 코드 · 3.2k / 16.2k | ModelScope 가중치 라이선스 [미확인] |
| 구형 기준선 | NeMo TitaNet-L / SpeechBrain ECAPA | 약 23M / 약 20M | 0.66 / 0.80 (Vox1-O) | – | CC BY 4.0 / Apache-2.0 · 18.8k / 12k | 2021-22년 결과 |

수치 출처: [arXiv 2510.04213](https://arxiv.org/html/2510.04213v2), [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2), [WeSpeaker](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2), [3D-Speaker](https://github.com/modelscope/3D-Speaker), [TitaNet card](https://huggingface.co/nvidia/speakerverification_en_titanet_large), [ECAPA card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). 표의 행마다 채점 설정(AS-Norm 유무 등)이 달라 소수점 차이는 큰 의미가 없다.

### 1-3초 구간에서는 EER이 세 배로 뛴다

VoxCeleb 0.1%대 차이는 GAMST 조건을 거의 설명하지 못한다. VoxCeleb은 깨끗한 영어 인터뷰이고 대부분 4초 이상이다. 짧은 구간에서는 좋은 모델도 크게 나빠진다. ECAPA-TDNN의 Vox1-O EER은 전체 길이 1.03%에서 **1초 3.04%**로 오른다 ([arXiv 2509.19721](https://arxiv.org/html/2509.19721v1)). ERes2NetV2는 2초에서 1.48%다 ([arXiv 2406.02167](https://arxiv.org/abs/2406.02167)). WeSpeaker는 LM 체크포인트가 주로 3초 이상 오디오에서 도움이 된다고 적었다 ([WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)). 한국어 SV 벤치마크는 공개 EER이 있는 것을 하나도 찾지 못했다. 유일한 한국어 연구(부산대, 2024)는 한국어 데이터를 더하면 모든 테스트셋에서 좋아졌다고만 쓰고 수치를 공개하지 않았다 ([KoreaScience](https://koreascience.or.kr/article/JAKO202414143303118.pub)). 따라서 여러 등록 클립의 중심 벡터, 구간 간 히스테리시스, 사내 한국어 dev 세트로 보정한 임계값이 모델 선택보다 중요하다.

### "화자 Y가 말한 키워드 X"는 캐스케이드로 만든다

이 과제를 한 모델로 푸는 오픈소스는 연구 코드뿐이다. 가장 잘 문서화된 것은 **ZP-KWS**(arXiv 2606.20106, 2026-09)다. 0.65M KWS 분기와 0.9M EfficientTDNN 화자 분기를 `p_final = p_utt · p_spk`로 곱한다. 등록은 발화 하나로 충분하다. target-only 모드에서 LibriPhrase-Easy EER 8.16%로 PK-MTL(17.74%)을 앞선다 ([arXiv 2606.20106](https://arxiv.org/html/2606.20106)). 하지만 저장소가 10 star이고 라이선스가 없으며 영어만 평가했다. PCOV-KWS, ProKWS, PK-MTL은 공개 코드를 확인하지 못했다. 결론은 성숙한 부품을 이은 캐스케이드다. VAD 구간을 Qwen3-ASR로 받아 쓰고, 키워드를 맞추고, ForcedAligner로 단어 시각을 잡는다. 화자 점수는 단어 자체(보통 1초 미만)가 아니라 그 단어를 둘러싼 VAD 구간에 매긴다. 두 점수는 ZP-KWS처럼 곱으로 합친다. 이 설계는 추론이며 아직 측정한 결과가 없다.

## 한국어 ASR: Qwen3-ASR-1.7B가 정확도와 단어 시각 모두 앞선다

한국어 수치를 공식 공개한 오픈 모델 중 최고는 **Qwen3-ASR-1.7B**다(Alibaba Qwen, 2026-01-29 공개, Apache-2.0, 3,941 star). **FLEURS-ko CER 2.57, Common Voice-ko 5.88, 대화체 MLC-SLM-ko 8.61**이다. 0.6B는 3.72 / 8.48 / 10.31이다 ([Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337)). star는 5k에 못 미치지만 공식 연구실 릴리스다. 같은 묶음의 **Qwen3-ForcedAligner-0.6B**는 한국어 단어 시각 오차(AAS)가 **37.2 ms**다. 300초 이어붙인 오디오에서도 42.2 ms다. 언어 평균으로 보면 NeMo NFA(129.8 ms)와 WhisperX(133.2 ms)보다 약 3배 정확하다. 긴 오디오에서는 WhisperX가 2708.4 ms까지 무너진다 ([Tech report Table 9](https://arxiv.org/html/2601.21337)). 키워드 횟수를 세는 데 40 ms 오차는 충분히 작다. 지연은 정렬이 아니라 ASR 창 크기에서 생긴다.

주의할 점이 하나 있다. Qwen3-ASR 스트리밍은 vLLM 백엔드에서만 돌고 타임스탬프를 내지 않는다 ([GitHub QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)). 스트리밍 성능 저하(1.7B 평균 WER 2.69 → 3.33)도 영어와 중국어로만 쟀다. 한국어 스트리밍 CER은 알 수 없다. 그래서 약 2초 롤링 창으로 받아 쓰고, 확정된 구간만 ForcedAligner로 정렬하는 "준스트리밍" 구조가 맞다.

진짜 저지연 스트리밍이 필요하면 **nemotron-3.5-asr-streaming-0.6b**(2026-06-04, OpenMDW-1.1, 상업 사용 허용)가 유일한 후보다. 80 ms부터 1.12 s까지 청크를 고를 수 있고 **FLEURS-ko CER 7.12–8.31**이다. 카드에는 단어 타임스탬프와 단어 부스팅 언급이 없다 ([HF card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)). 오프라인 대형 모델 **Nemotron 3.5 Transcribe 30B-A4B**는 FLEURS-ko 2.78로 Qwen3-ASR에 근접한다. 하지만 이 수치는 2차 요약 글에서 왔고 카드 원문은 확인하지 못했다 ([braindetox.kr 요약](https://braindetox.kr/en/posts/nvidia_nemotron_3_5_transcribe_korean_asr.html)). 2.57과 2.78은 두 회사가 각자 정규화한 결과라 직접 비교가 불완전하다. Canary와 Parakeet은 한국어를 지원하지 않는다 ([NeMo ASR checkpoints](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_checkpoints.html)). CPU·온디바이스용 최경량은 **sherpa-onnx Korean streaming zipformer**(79.02M, KsponSpeech 640 ms 청크 CER 9.91/10.72)다 ([icefall checkpoint](https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12)). 테스트셋이 달라 nemotron과 직접 비교할 수 없다.

| 역할 | 모델 | 한국어 CER | 단어 시각 | 바이어싱 | 라이선스 · star |
|---|---|---|---|---|---|
| 정확도 SOTA · 쓸 수 있는 SOTA | Qwen3-ASR-1.7B + ForcedAligner-0.6B | FLEURS 2.57 / CV 5.88 / MLC-SLM 8.61 | 37.2 ms AAS | system prompt context (수치 없음) | Apache-2.0 · 3,941 |
| 저비용 | Qwen3-ASR-0.6B + ForcedAligner | 3.72 / 8.48 / 10.31 | 동일 | 동일 | Apache-2.0 |
| 오프라인 대형 | Nemotron 3.5 Transcribe 30B-A4B | FLEURS 2.78 **[2차 출처]** | stream=false에서만 | "Context terms" (오삽입 경고) | OpenMDW-1.1 |
| 경량 스트리밍 | nemotron-3.5-asr-streaming-0.6b | FLEURS 7.12–8.31 | 언급 없음 | 언급 없음 | OpenMDW-1.1 · NeMo 18.8k |
| 생태계 | Whisper large-v3 (faster-whisper / WhisperX) | FLEURS 5.31 **[제3자, 미확인]** | WhisperX wav2vec2-ko, 한국어 오차 미공개 | `hotwords`, `initial_prompt` (수치 없음) | MIT 26,998 / BSD-2 25,658 |
| CPU 최경량 | sherpa-onnx Korean zipformer (2024) | KsponSpeech 9.91 / 10.72 | transducer 토큰 시각 (한국어 정확도 미확인) | hotwords 필드 있음, 한국어 미문서화 | Apache-2.0 · 16,215 |
| 보류 | SenseVoice-Small / Fun-ASR-MLT-Nano | 한국어 CER 미공개 | – | hotword 광고 | MIT 9,812 / Apache-2.0 1,616 |

수치 출처: [Qwen3-ASR report](https://arxiv.org/html/2601.21337), [nemotron streaming card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), [BuzzASR/korean](https://huggingface.co/BuzzASR/korean), [sherpa-onnx zipformer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html), [Fun-ASR-MLT-Nano card](https://huggingface.co/aoiandroid/Fun-ASR-MLT-Nano-2512).

### 한국어 키워드 바이어싱은 어디서도 측정되지 않았다

한국어에서 텍스트로 정의하는 오픈 어휘 KWS를 바로 쓸 수 있는 시스템은 없다. sherpa-onnx KWS 모델은 중국어와 영어뿐이다 ([sherpa-onnx KWS models](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html)). sherpa-onnx ASR hotwords는 transducer와 `modified_beam_search`에서만 작동한다. 문서화된 단위는 cjkchar와 bpe뿐이고 WER 개선 수치도 없다 ([sherpa-onnx hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)). Qwen3-ASR은 문맥 바이어싱 데이터로 학습했지만 핫워드 지표를 내지 않는다. Nemotron 카드는 문맥이 출력을 엉뚱한 용어로 끌어당길 수 있다고 경고한다. 그래서 가장 방어 가능한 설계는 키워드 목록을 문맥으로 넣고, 정규화된 전사 텍스트에서 센 뒤, ForcedAligner로 시각을 다는 방식이다. 한국어는 띄어쓰기 변형과 조사(키워드+가/를)를 처리해야 한다. 게이트는 키워드가 없는 발화에서의 오탐까지 재야 한다. 문맥 프롬프트가 오삽입을 만들 수 있기 때문이다.

## 자세 추정: 정확도 1위 Sapiens2는 생체정보 처리 금지 조항을 단다

2D 전신(손 포함) 정확도 1위는 Meta **Sapiens2**(ICLR 2026, 2026-04-24 공개)다. Meta 자체 308 키포인트 테스트에서 **0.4B 76.9, 1B 80.4, 5B 82.3 mAP**다. 같은 표에서 RTMW-X는 70.2, DWPose-L은 66.5, ViTPose+-H는 48.3이다 ([Sapiens2 paper](https://arxiv.org/html/2604.21681v1)). 5B는 1K 해상도 프레임당 15.72 TFLOPs이고 지연은 보고되지 않았다. 이 우위는 독립 검증이 안 됐다. COCO-WholeBody 수치가 없고, 논문 안에서 Sapiens-1B/2B 값이 표마다 다르다(76.8/78.3 대 74.1/75.3). 또 표 3의 RTMW-L/X 값(70.1/70.2)이 RTMW의 COCO-WholeBody 384×288 값과 똑같다. 기준선 행이 COCO 수치를 그대로 옮겨 왔을 가능성이 있다. 이 부분은 추론이며 확인하지 않았다. 더 큰 문제는 라이선스다. Sapiens2 License는 코드와 가중치 모두에 적용되고, "surveillance"와 "biometric processing" 용도를 금지하며, Meta의 감사와 삭제 요구 권한을 둔다 ([Sapiens2 LICENSE.md](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)). 특정 스트리머를 추적하는 GAMST 용도가 여기에 걸리는지는 법률 검토가 필요하다. 저장소는 1,049 star의 공식 Meta 릴리스다. Sapiens v1(5,440 star)은 CC BY-NC 4.0이라 상업 사용이 안 된다.

쓸 수 있는 SOTA는 **RTMW-x 384×288**이다. COCO-WholeBody Whole AP 70.2, 29.3 GFLOPs다 ([RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)). mmpose(8,080 star)와 rtmlib(705 star, 2026-11 활성)는 모두 Apache-2.0이다. mmpose는 2025-08 이후 push가 없어 유지보수가 느려지고 있다. 그래서 rtmlib ONNX 경로가 낫다. top-down 전신 모델은 손을 빠뜨리지 않는다. 반면 MediaPipe Holistic은 수어 영상에서 왼손/오른손이 **20.22/23.61%** 프레임에서 빠졌다 ([arXiv 2604.24609](https://arxiv.org/html/2604.24609)). 손 모양이 중요한 "샷건" 제스처에는 이 차이가 결정적이다. 견고성이 필요하면 **SDPose**(COCO-WholeBody 72.8 AP, MIT 코드)가 있다. 하지만 181 star이고 V100에서 0.84 fps로 가장 느렸다 ([SDPose paper](https://arxiv.org/html/2509.24980v2); [arXiv 2604.24609](https://arxiv.org/html/2604.24609)).

경량 스트리밍은 **RTMPose-m**(COCO 75.8 AP, GTX 1660Ti TRT-FP16 2.29 ms, i7 CPU 11.06 ms)과 **MediaPipe Pose Landmarker**(Apache-2.0, 37,840 star)다. MediaPipe만 내장 VIDEO/LIVE_STREAM 추적 모드가 있다 ([Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)). 2026년 독립 비교에서 지터도 가장 낮았다(jerk 2.46/6.51, Sapiens는 7.77/10.86). **YOLO26-pose**는 COCO-17 속도 대비 정확도가 가장 좋다(x 71.6 mAP, T4 12.2 ms). 하지만 **AGPL-3.0**이고 손이 없다 ([Ultralytics docs](https://docs.ultralytics.com/tasks/pose/)). Apache-2.0 SDK에는 기본 의존성으로 넣으면 안 된다. 사용자가 고르는 선택 extra로만 둔다. OpenPose(34.6k star)는 학술·비영리 전용이다.

| 역할 | 모델 | 벤치마크 | 속도 | 라이선스 · star | 플래그 |
|---|---|---|---|---|---|
| 2D 정확도 SOTA | Sapiens2-1B / 5B | 308-kp 80.4 / 82.3 mAP (Meta 자체) | 미보고 (5B 15.72 TFLOPs) | Sapiens2 License · 1,049 | **biometric processing·감시 금지**, 감사 조항 |
| 견고성 | SDPose | COCO-WB 72.8 AP | 0.84 fps V100 | MIT 코드 · 181 | SD2 가중치 라이선스 [미확인] |
| 2D 쓸 수 있는 SOTA | RTMW-x 384×288 (rtmlib) | COCO-WB 70.2 AP | 29.3 GFLOPs | Apache-2.0 · 8,080 / 705 | mmpose 정체 |
| 경량 스트리밍 | RTMPose-m / RTMW-l | COCO 75.8 / COCO-WB 70.1 | 2.29 ms GPU, 11.06 ms CPU | Apache-2.0 | – |
| 경량·추적 내장 | MediaPipe Pose Landmarker | 공개 정확도 미수집 | 지터 최저 | Apache-2.0 · 37,840 | 손 누락 20-24% |
| 선택 extra | YOLO26-pose | COCO-17 71.6 (x) | 12.2 ms T4 | **AGPL-3.0** · 63,723 | 몸 17점만 |
| 사용 금지 | Sapiens v1 / OpenPose | – | – | **CC BY-NC** / **비상업 연구 전용** | – |

### 3D는 계수에 꼭 필요하지 않다

단일 이미지 3D 정확도는 **Fast SAM 3D Body**가 앞선다. 2026-11 동일 프로토콜 재평가에서 PA-MPJPE가 EMDB-1 38.6, 3DPW 34.1, RICH 32.1 mm다. 하지만 사람당 130 ms다(원본 SAM 3D Body는 416 ms) ([InstantHMR arXiv 2611.09195](https://arxiv.org/abs/2611.09195)). SAM License로 상업 사용이 가능하고 3,766 star의 공식 Meta 릴리스다 ([SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)). 실시간 3D의 쓸 수 있는 SOTA는 **NLF-L**이다. 39.3/35.1/27.3 mm로 SAM 3D Body와 거의 같고 RTX 4070 TRT fp16에서 **4.32 ms**다. NLF-S는 1.52 ms다. 코드는 MIT(471 star)다. 다만 SMPL 계열 출력의 모델 파일 라이선스는 확인하지 않았다. InstantHMR-M은 0.98 ms로 가장 빠르지만 기본 가중치가 비상업이다. 상업용 M-C 변형은 EMDB-1에서 72.2 mm로 크게 나빠진다. 손 전용으로는 WiLoR가 HaMeR보다 정확하고(FreiHAND PA-MPJPE 5.5 대 6.0) 지터도 훨씬 낮다(5.92 대 20.43). 하지만 **CC BY-NC-ND**다 ([WiLoR paper](https://arxiv.org/html/2409.12259v2)). HaMeR는 MIT다. 영상 3D인 GVHMR은 상업 사용 금지이고, WHAM은 2024-04 이후 정체다. 제스처 계수에는 팔꿈치·어깨 각도 같은 시점 불변 관절 특징이면 충분하다. MediaPipe world landmarks나 NLF로 얻을 수 있다. 전체 메시는 지연만 늘린다.

## 원샷 제스처 계수: 5k-star 제품은 없고, 자세 파이프라인을 직접 조립한다

논문상 RepCount-A 1위는 **PAMS / Count What Repeats**(CVPR 2026 Findings)다. 골격 기반 자기지도 모델로 **MAE 0.129, OBO 0.842**를 낸다. 교차 데이터셋에서도 같은 수준을 유지한다 ([Count What Repeats PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)). 다음은 CountLLM(CVPR 2025, 0.162/0.639, Vicuna-7B)이다 ([CountLLM arXiv](https://arxiv.org/abs/2503.17690)). 둘 다 공개 코드를 찾지 못했다. 더 근본적인 문제도 있다. PAMS, D2-STX, RepNet, TransRAC는 클립에서 가장 두드러진 반복을 센다. 사용자가 "이 동작을 세라"고 지정할 수 없다. GAMST의 owner-declared Watch에는 이 대상 선택성이 핵심이다.

예시를 입력으로 받는 유일한 공개 모델은 **ESCounts**(ACCV 2024, Bristol, MIT, 32 star)다. RepCount에서 예시 없이 0.213/0.563이다. 같은 영상의 예시 1개를 주면 0.210/0.589다 ([ESCounts arXiv](https://arxiv.org/abs/2403.18074)). 예시가 주는 이득은 작다. 게다가 피트니스 클래스로 학습해서, 스트리머 고유 제스처에서 예시 조건이 통하는지는 검증된 적이 없다. 2026-10 독립 벤치마크 RepProbe에서는 ESCounts가 가장 강한 전문 계수기였다(절대 MAE 1.07, OBO 77.5%). 제로샷 VLM은 크게 뒤졌다. Qwen3-VL-8B는 MAE 5.02, OBO 35.0%였다. 합성 분할의 고횟수 구간에서는 모든 범용 VLM이 OBO 0%였다 ([RepProbe arXiv](https://arxiv.org/abs/2610.18844)). VLM에 프레임을 넣어 세게 하지 말라는 GAMST 원칙을 수치가 뒷받침한다. D2-STX(CVPR 2026 Findings, MIT, 1 star)는 10–20 ms 추론을 주장한다. 하지만 ESCounts ViT-B(0.213)를 이기지 못한다(0.255). D2-STX의 SOTA 주장은 같은 Swin-T 백본끼리 비교할 때만 성립한다 ([D2-STX PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf)).

진짜 학습 없는 경로는 신호처리뿐이다. 골격 자기유사도와 스펙트로그램을 쓴 Hsu et al.(2021)은 깨끗한 운동 데이터 MM-Fit에서 **MAE 0.06, OBOA 0.94**를 냈다 ([arXiv 2107.13760](https://arxiv.org/abs/2107.13760)). 하지만 RepCount에서는 돌린 적이 없어 직접 비교할 수 없다. 반대 증거도 있다. D2-STX의 자세 단독 분기는 실환경 RepCount에서 MAE 0.843으로 무너졌다. 저자들은 주변 인물과 가림 때문에 자세 추정이 실패했다고 설명한다. 단일 인물이 정면을 보는 스트리머 화면은 Hsu et al. 조건에 더 가깝다. 다만 이것은 추론이다. DTW 계열 원샷 매칭(uDTW)은 NTU-120 원샷에서 49.0%로, 최신 SkelHCC(71.6%)에 뒤진다 ([SkelHCC arXiv](https://arxiv.org/abs/2606.03610)). SkelHCC 코드는 아직 공개가 확인되지 않았다. ≥5k star 프로젝트(Ultralytics, MediaPipe, MMPose)는 자세 추정기일 뿐이다. 예시 기반 계수기는 아니다. 찾은 반복 계수 저장소는 모두 120 star 이하였다.

| 역할 | 방법 | RepCount-A MAE / OBO | 예시 입력 | 코드 · 라이선스 · star | 비고 |
|---|---|---|---|---|---|
| 논문 정확도 SOTA | PAMS (CVPR'26 F) | 0.129 / 0.842 | 없음 | **코드 미공개** | 자체 보고, 대상 지정 불가 |
| 2위 | CountLLM (CVPR'25) | 0.162 / 0.639 | 텍스트 프롬프트 | **코드 미공개** | 7B, 학습 8×A100 |
| 쓸 수 있는 예시 SOTA | ESCounts ViT-B | 0.213 / 0.563 (1-shot 0.210 / 0.589) | 영상 예시 0–2개 | MIT · 32 | 체크포인트 공개 [미확인], GPU |
| 경량 학습 모델 | D2-STX (Swin-T) | 0.255 / 0.500 | 없음 | MIT · 1 | 10–20 ms 자체 주장 |
| 권장 경량 경로 | 자세 + 예시 유사도 + 피크/DTW | RepCount 수치 없음 (MM-Fit MAE 0.06) | 예시 1개 | 직접 구현 (Apache-2.0 부품) | 대상 지정 가능, 학습 없음 |
| 사용 금지 | 범용 VLM 제로샷 | RepProbe MAE 5.02 (Qwen3-VL-8B) | 프롬프트 | – | 고횟수에서 붕괴 |

권장 구조는 PAMS의 정신을 따른다. 학습된 주기 임베딩 대신 예시 유사도를 쓴다. 먼저 RTMW(손 필요 시) 또는 MediaPipe로 키포인트를 얻는다. One-Euro 필터로 지터를 줄인다. One-Euro는 InstantHMR에서 지연 없이 가속 오차를 27.1에서 15.2로 줄였다 ([InstantHMR](https://arxiv.org/abs/2611.09195)). 다음으로 루트 중심·스케일 정규화를 하고, 예시의 핵심 자세와 코사인/OKS 유사도를 1D 신호로 만든다. 마지막으로 다중 스케일 피크 검출과 히스테리시스로 센다. "샷건"처럼 반복되는 제스처에는 이 방식이 맞다. 축구·농구 패스처럼 드문 비주기 이벤트에는 FFT 계수가 맞지 않는다. 예시 궤적에 대한 부분수열 DTW 검출이 맞다. 이 설계 전체는 추론이며 벤치마크된 결과가 아니다. ESCounts는 GPU 쪽 비교 기준선으로 둔다.

## 확인하지 못한 항목

아래 항목은 근거가 2차 출처, 검색 스니펫, 기억뿐이거나 아예 확인하지 않았다. 채택 전에 확인해야 한다.

| # | 항목 | 영향 |
|---|---|---|
| 1 | VoxBlink2, CN-Celeb 데이터셋 라이선스(비상업 CC BY-NC-SA라는 기억) | ReDimNet2 vb2/cnc2, SimAM-VB2, ResNet293-VB2 가중치의 상업 사용 |
| 2 | 3D-Speaker ModelScope 가중치(CAM++, ERes2NetV2)의 모델 카드별 라이선스 | 경량 SV 기본값 |
| 3 | w2v-BERT 2.0 기반 모델 자체의 라이선스, WeSpeaker w2v-BERT 체크포인트 배포 조건 | 정확도 SOTA 서버 재채점 |
| 4 | "Beyond Short Segments"(arXiv 2609.25007) 1초 EER 20.449 → 14.571%는 스니펫만 봄 | 짧은 구간 기대치 |
| 5 | 한국어 SV 공개 벤치마크 없음. AIHub 한국어 화자 코퍼스 존재·조건 미확인 | 평가 세트 구축 |
| 6 | 화자 검증 자체가 생체정보 처리에 해당하는지에 대한 한국 법적 해석은 조사하지 않음 | SV 기능 출시 |
| 7 | Nemotron 3.5 Transcribe FLEURS-ko 2.78, 타임스탬프 조건은 2차 요약 글 기준 | ASR 2순위 |
| 8 | Whisper large-v3 한국어 CER(FLEURS 5.31, CV 6.01, KsponSpeech 약 12%)은 제3자 카드 | Whisper 비교 |
| 9 | TeamUNIVA가 보고한 Qwen3-ASR-1.7B FLEURS-ko CER 1.42는 공식 2.57과 충돌(정규화 차이 추정) | 한국어 정확도 해석 |
| 10 | Qwen3-ASR 한국어 스트리밍 CER, 모든 모델의 한국어 바이어싱 효과(B-WER, 키워드 recall) 미공개 | 키워드 계수 게이트 |
| 11 | sherpa-onnx 한국어 hotwords 동작(bpe 토큰 직접 입력 시)은 미문서화 | CPU 경로 |
| 12 | Moonshine, Kimi-Audio, Step-Audio의 한국어 지원·CER, 화자 조건 ASR(TS-ASR) 미조사 | 후보 누락 가능 |
| 13 | Sapiens2 표 불일치(표 3 대 표 8, 테스트셋 크기), 기준선 RTMW 값이 COCO-WB 값과 동일 | 2D SOTA 우위의 크기 |
| 14 | Sapiens2 지연·FPS 미보고. "biometric processing" 금지가 제스처 계수에 적용되는지 법적 해석 없음 | Sapiens2 채택 여부 |
| 15 | SDPose 가중치(SD2 U-Net)의 라이선스, ViTPose·RTMW 가중치가 코드 라이선스와 같은지 | 2D 대안 |
| 16 | SMPL/SMPL-X 모델 파일 라이선스(NLF, HMR2.0 등 출력), MHR이 Apache-2.0이라는 정보는 스니펫뿐 | 3D 경로 |
| 17 | ESCounts, D2-STX 학습 체크포인트 공개 여부 | 계수 기준선 |
| 18 | PAMS 수치는 자체 보고이며 재현 없음. PAMS, CountLLM, SkelHCC 코드 미공개 | 계수 SOTA 판단 |
| 19 | Ultralytics AIGym이 관절 각도 임계값 방식이라는 점은 이번에 재확인하지 않음 | 5k-star 대안 평가 |
| 20 | 반복 계수 GitHub 검색은 "repetition counting" 한 쿼리만 사용 | 5k-star 계수기 부재 결론 |
| 21 | 모든 벤치마크가 피트니스·Kinetics 같은 밀집 주기 동작. 스트리머 제스처·패스 같은 희소 동작 평가는 없음 | 계수 정확도 기대치 |

## 결론

세 과제에서 공통으로 보이는 사실이 있다. 남은 정확도 차이는 모델 크기가 아니라 도메인과 라이선스가 결정한다. 화자 검증에서 CN-Celeb 데이터를 넣으면 EER이 7.49%에서 3.59%로 떨어진다. VoxCeleb 상위권끼리의 차이보다 몇 배 크다. 한국어 ASR은 오픈 모델이 이미 FLEURS-ko CER 2-3%대에 와 있다. 이제 병목은 전사가 아니라 조사 처리, 신조어, 바이어싱 오삽입 같은 계수 규칙이다. 자세 추정에서는 정확도 1위 두 모델(Sapiens2, SAM 3D Body)이 모두 Meta 맞춤 라이선스를 단다. Apache-2.0 SDK의 기본값은 한 세대 아래인 RTMW와 MediaPipe가 맡을 수밖에 없다. 그래서 GAMST가 차별화할 지점은 "더 큰 모델"이 아니다. 한국어 스트림 1-3초 SV 시험 목록, 소유자가 녹음한 키워드 세트, 스트리머 제스처 예시 세트를 만들어, 후보 모델을 같은 지표 게이트로 직접 재는 일이다.

또 하나 눈여겨볼 점이 있다. 예시 한 개로 사용자가 지정한 동작을 세는 오픈소스는 큰 프로젝트 중에 아직 없다. 독립 벤치마크는 범용 VLM이 이 일을 못 한다는 것을 보여 준다(고횟수 OBO 0%). 고빈도 지각은 모델 밖에서 처리하고 VLM에는 타임스탬프가 달린 증거만 넘긴다는 GAMST 설계가 바로 이 빈자리에 맞는다. 자세 키포인트, 예시 유사도, 피크/DTW로 만든 계수기를 RepCount와 사내 제스처 세트로 재서 공개하면, 이 분야에서 측정치를 갖춘 첫 Apache-2.0 구현이 될 수 있다.
