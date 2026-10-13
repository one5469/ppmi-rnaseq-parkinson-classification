# PPMI RNA-seq 기반 파킨슨병 분류 및 차등발현 분석

PPMI RNA-seq 데이터를 활용해 파킨슨병(PD)과 대조군의 유전자 발현 차이를 분석하고, TabNet 기반 분류와 자기지도 사전학습을 실험한 프로젝트입니다. 집단 구성과 재표본추출에 따른 차등발현 결과의 안정성도 함께 분석했습니다.

## 주요 작업

- Raw count·TPM과 대상자 메타데이터의 샘플 ID 정리 및 병합
- 진단군·방문·QC 조건에 따른 데이터 필터링
- PyDESeq2 기반 차등발현과 K-fold·Bootstrap·무작위 추출 비교
- 유의 유전자 집합의 유사도·검출 빈도·발현 변화 방향 등 분석
- 환자 단위 교차검증 및 학습 fold에서의 유전자 선택
- TabNet 분류와 선택 가능한 baseline 모델 비교
- 검증 환자를 사전학습에서도 제외하는 TabNet 자기지도 학습
- 사전학습 encoder 표현과 원본 입력의 분류 비교
- 사전학습 모델의 fine-tuning과 encoder 고정 옵션 구성

## 데이터와 기술

- 데이터: PPMI RNA-seq IR3 raw count, TPM, 대상자 메타데이터
- 중심 분류 과제: PD–Control; 일부 차등발현 노트북은 Genetic Cohort–Control 설정
- 기술: Python, PyDESeq2, pytorch-tabnet, PyTorch, scikit-learn

## 코드 안내

노트북 파일명과 위치를 유지했습니다. 원하는 분석의 파일을 선택해 설정 셀부터 확인하면 됩니다.

| 파일 | 내용 |
| --- | --- |
| [tabnet_classifier_RNA_cv_baselines_DESeq_feature_selection_fixed_v2.ipynb](tabnet_classifier_RNA_cv_baselines_DESeq_feature_selection_fixed_v2.ipynb) | TabNet 분류 및 fold별 DESeq 유전자 선택·baseline 비교 |
| [ppmi_pydeseq2_stability_analysis_v0412_random_sampling.ipynb](ppmi_pydeseq2_stability_analysis_v0412_random_sampling.ipynb) | 비복원 무작위 추출을 포함한 안정성 분석 |
| [tabnet_classifier_RNA_cv_encoder_backbone_experiments_fixed_prefilter_fs.ipynb](tabnet_classifier_RNA_cv_encoder_backbone_experiments_fixed_prefilter_fs.ipynb) | 사전학습 encoder 표현과 원본 입력의 분류 비교 |
| [ppmi_pydeseq2_full_vs_kfold_train_test_DE_03921_v0412.ipynb](ppmi_pydeseq2_full_vs_kfold_train_test_DE_03921_v0412.ipynb) | 전체 데이터와 K-fold 학습·테스트 부분집합의 차등발현 비교 |
| [tabnet_classifier_RNA_cv_ssl_finetune_DESeq_configurable_freeze_option_prefilter_fs.ipynb](tabnet_classifier_RNA_cv_ssl_finetune_DESeq_configurable_freeze_option_prefilter_fs.ipynb) | scratch·사전학습 fine-tuning 및 encoder freeze 설정 |
| [ppmi_pydeseq2_stability_analysis_v0412.ipynb](ppmi_pydeseq2_stability_analysis_v0412.ipynb) | K-fold·Bootstrap 차등발현 안정성 분석 |
| [ppmi_pydeseq2_stability_analysis.ipynb](ppmi_pydeseq2_stability_analysis.ipynb) | 차등발현 안정성 분석 초기 버전 |
| [ppmi_tabnet_ssl_pretrain.py](ppmi_tabnet_ssl_pretrain.py) | fold별 TabNet 자기지도 사전학습 및 encoder 저장 |

설치·데이터 준비·실험 선택은 [docs/SETUP.md](docs/SETUP.md)에 정리했습니다.

노트북 출력과 실행 번호는 공개용으로 정리했고, 코드 셀은 원본과 동일합니다. 실제 PPMI 데이터와 학습 가중치는 포함하지 않았습니다. 각 파일의 기본 설정이 다르며, 모든 비교 조건이 실행되었다는 의미는 아닙니다. 최종 성능 수치는 별도 실험 기록을 기준으로 추가할 수 있습니다.
