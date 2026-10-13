# 환경·데이터·실험 선택

## 설치와 작업 위치

저장소 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Python 3.9 이상 문법을 사용하는 소스가 포함되어 있습니다. 패키지 목록은 import를 기준으로 작성했고 원래 환경의 버전은 제공되지 않아 고정하지 않았습니다. PyDESeq2·TabNet API 호환성과 GPU 패키지는 본인의 기존 실험 환경에서 확인해야 합니다. 이번 정리에서는 원 데이터가 없어 전체 분석·학습을 다시 실행하지 않았습니다.

## 입력 파일

| 저장소 루트 기준 경로 | 내용 |
| --- | --- |
| `Dataset/RNA/IR3/counts/` | featureCounts raw count 파일 |
| `Dataset/RNA/IR3/quant/tpm_gene_matrix.csv` | 기존 형식의 TPM 행렬 |
| `Dataset/RNA/IR3/metaDataIR3.csv` | 샘플·환자·진단·방문·QC 메타데이터 |

TPM 노트북은 `index_col=0`으로 파일을 읽으며 ENSG/ENST 형태의 열 이름을 유전자 feature로 사용합니다. 기존 샘플 ID 매칭 방식에 맞는 행렬을 준비하세요. 환자 ID는 `PATNO_norm`을 사용하고, 차등발현 코드에서는 `HudAlphaID`, `DIAGNOSIS_GROUP`, `QCflagIR3` 등의 메타데이터를 사용합니다. ID 정규화·복구와 진단·방문 열 탐지는 소스에 구현되어 있습니다.

일부 설정에서는 유전자 annotation CSV도 필요합니다. Fine-tuning 노트북의 현재 기본값은 `GENE_ANNOTATION_PATH='gencode_v29_gene_annotation_pri.csv'`, `GENE_BIOTYPE_FILTER='protein_coding_lncRNA'`입니다. 이 파일은 첨부되지 않았습니다. 기존 annotation을 해당 위치에 두거나, 설정 셀의 annotation·prefilter 조건을 본인의 실험 설계에 맞게 지정해야 합니다. Encoder 비교 노트북의 기본 annotation 경로는 `None`입니다.

## 실험 선택

1. **차등발현·안정성**: `ppmi_pydeseq2_*.ipynb`에서 `AnalysisConfig`의 데이터 경로, 진단군, 방문, 공변량 및 표본추출 방식을 확인합니다. 전체·K-fold 비교 노트북의 기본 진단군은 Genetic Cohort–Control이므로 PD–Control 분석 시 설정을 바꾸세요.
2. **TabNet scratch·baseline**: `tabnet_classifier_RNA_cv_baselines_DESeq_feature_selection_fixed_v2.ipynb`에서 실행합니다. `BASELINE_MODELS`가 `None`이면 baseline을 실행하지 않습니다. DESeq 유전자 선택 여부와 외부 결과 재사용 설정을 확인하세요.
3. **자기지도 사전학습**: 아래 Python 스크립트로 fold별 encoder와 관련 정보를 저장합니다.
4. **Encoder 입력 비교**: `...encoder_backbone_experiments_fixed_prefilter_fs.ipynb`의 `PRETRAINED_TABNET_RUN_DIR`을 실제 사전학습 결과로 바꿉니다. 현재 기본 설정은 pretrained 모델을 사용하므로 기존 경로 그대로 실행하면 파일이 없을 수 있습니다.
5. **Fine-tuning·freeze**: `...ssl_finetune_DESeq_configurable_freeze_option_prefilter_fs.ipynb`에서 `TABNET_TRAIN_MODE`를 `scratch`, `pretrained`, `both` 중 선택하고 `FREEZE_PRETRAINED_ENCODER` 등을 확인합니다. 사전학습을 사용할 때는 결과 경로와 feature·scaler 파일 설정을 함께 확인합니다.

## 자기지도 사전학습 명령

```bash
python ppmi_tabnet_ssl_pretrain.py --help
```

기존 기본값을 사용하는 학습 예시입니다. GPU를 사용할 경우 학습 환경에 맞게 지정하세요.

```bash
CUDA_VISIBLE_DEVICES=0 python ppmi_tabnet_ssl_pretrain.py   --tpm-path ./Dataset/RNA/IR3/quant/tpm_gene_matrix.csv   --meta-path ./Dataset/RNA/IR3/metaDataIR3.csv   --target-groups PD Control   --n-splits 5   --out-dir ./results_tabnet_rna_ssl_pretrain_only
```

후속 노트북은 저장 결과 중 `tabnet_pretrainer_full.zip`, feature 목록과 scaler 등을 설정에 따라 읽습니다. 원본의 사전학습 결과 경로와 노트북의 하드코딩된 결과 경로는 자동으로 연결되지 않습니다. 실제 실행 폴더로 직접 지정해야 합니다.

이 파일들은 하나의 자동 실행 파이프라인으로 재작성하지 않고 기존 독립 실험 파일을 유지한 것입니다. Notebook 출력은 제거했으므로 결과 표·그림은 데이터와 설정을 준비한 후 다시 실행하면 생성됩니다.
