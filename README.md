# WR227 - Technical Writing Project

- Group 6
  - 24125049 - Vong Chi Van
  - 24125041 - Pham Nguyen Minh Quan

## Project Description

- Topic: Improve Vision Transformer's Time Performance via Linear Attention and Token Pruning

## Progress

- [x] [PA1](https://docs.google.com/document/d/1EMJ5UtmZ4s529wdfB6jzrk_XtRUov0Cv9JmKBHgcT_I/): Topic Proposal
- [x] PA2: Introduction, Related Work
- [x] PA3: Method
- [x] PA4: Experiment and Results
- [x] PA5: Conclusion and Abstract
- [ ] In-class Presentation

## Material

- [Google Drive](https://drive.google.com/drive/folders/1JU3nthPxffqqXxcfkusvtmB67QRu5dmF)
- [Literature Review](https://docs.google.com/spreadsheets/d/13qwilvCUy6kqlFihILLLDaHHqWTQjU1UKLHDgjB9DzE/)

## To-do List

- [x] More literature survey
- [x] Implement Transformer architecture
- [x] Vision Transformer
- [x] Format: 2 columns
- [x] Refine introduction section
- [x] Revise RL method: 2 states keep and prune
- [x] Implement other standard token pruning methods
- [x] Preprocess and analyze another dataset to avoid overfitting
- [x] Benchmark and analyze performance of the models
- [ ] Implement object detection

- introduction: ViT time -> different approaches [citations] -> linear -> underperform
  -> different approaches linear [citations] -> rl-based token pruning -> combine linear + rl with appropriate reward function.
- experiment:
  - train other models on our choosen dataset and compare
  - benchmark: a metric for time-performance (param, flops) trade-off
- ablation study: remove components of our model (can substitute with other comparable components)
- quotation marks
- refer to figures, tables
- the disccusion about figures, tables should be in the same page
- graph must use the same font as text

## Feedback

- Perform T-test
- Make the motivation clearer
- Abstract the attention mechanism (Transformer, ViT are just candidates), view the current work as framework that can be applied for different model using attention.
- Move the background (linear attention, softmax attention) into the method
- Related work of linear attention should represent current papers that improve linear attention
