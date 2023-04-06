# MAGRec (Multi-domAin Graph-based Recommender)

This is our Pytorch implementation of the paper:

Ariza-Casabona, A., Twardowsky, B., Wijaya, T. K. (2023). [Exploiting Graph Structured Cross-Domain Representation for Multi-Domain Recommendation](https://arxiv.org/pdf/2302.05990.pdf), ECIR23.

Please cite our paper if you use this repository:

````
@inproceedings{arizacasabona2023magrec,
  author    = {Alejandro Ariza{-}Casabona and
               Bartlomiej Twardowski and
               Tri Kurniawan Wijaya},
  title     = {Exploiting Graph Structured Cross-Domain Representation for Multi-domain
               Recommendation},
  booktitle = {Advances in Information Retrieval - 45th European Conference on Information
               Retrieval, {ECIR} 2023, Dublin, Ireland, April 2-6, 2023, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13980},
  pages     = {49--65},
  publisher = {Springer},
  year      = {2023},
  url       = {https://doi.org/10.1007/978-3-031-28244-7\_4},
  doi       = {10.1007/978-3-031-28244-7\_4}
}
````

## Datasets

For our paper, we used the 2018 version of the Amazon reviews dataset that can be downloaded [here](https://nijianmo.github.io/amazon/index.html) or using the **download_data.py** script. To prepare the dataset (this process can be skipped as it also is performed by the code pipeline if the preprocessed dataset is missing) for any combination of domains a.k.a. categories, use the **preprocess_data.py** script. Modify the data paths accordingly in the **utils/constants.py** file.

An illustration of the amount of domain overlapping among any domain pair of the ones selected for the paper is presented below (Domain names are shortened by keeping the first letter of each word e.g. Arts Crafts and Sewing (ACaS), Luxury Beauty (LB), etc):

<img src="https://github.com/alarca94/magrec/blob/master/images/domain_user_overlap.png" width="750">

The dataset partitions of our paper are:

* Amazon-2a: ACaS, PLaG
* Amazon-2b: OP, TaG
* Amazon-2c: OP, PLaG
* Amazon-2d: TaG, VG
* Amazon-3: OP, PP, S
* Amazon-6: MI, OP, PLaG, PP, TaG, VG
* Amazon-13: All domains included in the previous image

## Code

To train and evaluate MAGRec (using the first CUDA device), run the following command:

```
CUDA_VISIBLE_DEVICES=0 python run_mdr.py --model-name MAGRec --model-variant MemGNN --domains "Toys and Games,Video Games" --graph-type interacting --no-use-hyperopt
```

Replace the given domains and the graph type (disjoint/flattened/interacting) as needed.

## Dependencies

The following dependencies were used during our experiments:

````
- pandas==1.1.5
- numpy==1.19.5
- hyperopt==0.2.5
- deepctr-torch
- torch==1.8.1
- tqdm
- tensorflow==1.12
- torch-geometric==2.0.3
- (optional) ray[TUNE]
````
