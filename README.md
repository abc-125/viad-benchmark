# Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection

## Work in progress

✨[Spotlight paper](https://arxiv.org/abs/2503.23451)✨ at VAND workshop at CVPR 2025. It presents a comprehensive empirical analysis of Visual Industrial Anomaly Detection with a focus on real-world applications. We demonstrate that recent SOTA methods perform worse than methods from 2021 when evaluated on a variety of datasets. We also investigate how different practical aspects, such as input size, distribution shift, data contamination, supervised training, and having a validation set, affect the results.

**TLDR**:
- Existing academic evaluation practices do not reliably predict real-world industrial performance for anomaly detection models (e.g., early stop on the test dataset, center crop on the object).
- Higher input resolution helps detect small defects, but for some models, it also makes the detection of large and logical defects worse, demonstrating the restrictions of the receptive field.
- Anomaly detection models are sensitive to noisy labels in the training data, and truly unsupervised approaches still underperform compared to one-class methods even in the presence of noisy labels.
- Data distribution shift is detected as anomaly by all the tested models, demonstrating the necessity for models which are capable of capturing semantic meaning rather than pixel values.
- Usage of the validation set for an early stop strongly improves performance, even if it is not identical to the test set.


## 🔧 Install
```bash
git clone https://github.com/abc-125/viad-benchmark
cd viad-benchmark/anomalib
pip install -e .
anomalib install
cd ..
pip install mlflow==2.21.3
```

## 🚀 Get started
Modify `run.py` to include required models and datasets. Currently, only the Evaluation experiment is available.
```python
python run.py
```

## 🙏 Acknowledgement

We use [Anomalib](https://github.com/open-edge-platform/anomalib) library and the official implementations of models: [DRAEM](https://github.com/VitjanZ/DRAEM), 
[MMR](https://github.com/zhangzilongc/MMR), [MSFlow](https://github.com/cool-xuan/msflow), [GLASS](https://github.com/cqylunlun/GLASS),
[SimpleNet](https://github.com/DonaldRR/SimpleNet), [DevNet](https://github.com/GuansongPang/deviation-network), 
[DRA](https://github.com/Choubo/DRA), [InReaCh](https://github.com/DeclanMcIntosh/InReaCh). We are thankful for their amazing work!


## ✨ Citation
Please cite our paper if you find it useful:

```
@inproceedings{baitieva2025benchmark,
      title={Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection}, 
      author={Aimira Baitieva and Yacine Bouaouni and Alexandre Briot and Dick Ameln and Souhaiel Khalfaoui and Samet Akcay},
      booktitle = {CVPRW},
      year={2025},
      pages={4024-4034}
}
```
