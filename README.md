
<!---

    Copyright (c) 2021 Robert Bosch GmbH

-->

# To Share or not to Share:

This is the companion code for the experiments reported in the paper

> "To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning" by Lukas Lange, Jannik Strötgen, Heike Adel and Dietrich Klakow published at EMNLP 2021.

The paper can be found [here](https://arxiv.org/pdf/2104.08078.pdf). The code allows the users to reproduce the results reported in the paper and extend the model to new datasets and embedding configurations. 
Please cite the above paper when reporting, reproducing or extending the results as:

## Citation

```
@inproceedings{lange-etal-2021-share,
    title = "To Share or not to Share: {P}redicting Sets of Sources for Model Transfer Learning",
    author = {Lange, Lukas  and
      Str{\"o}tgen, Jannik  and
      Adel, Heike  and
      Klakow, Dietrich},
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.689",
    pages = "8744--8753",
}
```

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "To Share or not to Share: Predicting Sets of Sources for Model Transfer Learning". 
It will neither be maintained nor monitored in any way.

## Setup

* Install PyTorch, transformers, nltk, scipy, sklearn (tested with Huggingface>=4, PyTorch 1.3.1 and Python 3.6.8).
* Download pre-trained BERT model.
* Prepare corpora in BIO format.
* Train one of our sequence labeling models.

## Data

We do not ship the corpora used in the experiments from the paper. 
Please follow the references and descriptions in the paper for more information.  

## Experiments
The domain transfer experiments consist of 2 training steps:

Use `-t` for the task (NER/GUM/POS/TIME), e.g., `-t NER` and `-d` for the specific dataset, e.g., `-d Financial`. You can also abbreviate this call using the first character of the task and dataset, i.e., `-t n -d f`

1. **NER** (different labels): News *(CoNLL 2003)*, Wetlab *(WNUT-20)*, Social *(WNUT-17)*, Twitter *(WNUT-16)*, Privacy *(i2b2 2014 anonymization)*, Clinical *(i2b2 2010 extraction)*, Financial *(SEC)*, Literature *(LitBank)*, Materials *(SOFC-Exp)*
2. NER (w/ **GUM**, 11 labels): Academic, Biography, Fiction, Interview, News, Reddit, Voyage, Wikihow
3. **POS** (w/ GUM, 17 labels): Academic, Biography, Fiction, Interview, News, Reddit, Voyage, Wikihow
4. **Time** Tagging (4 labels): Discussion *(ACE05)*, Telephony *(ACE05)*, Broadcast-News *(ACE05)*, Newswire *(ACE05)*, Usenet *(ACE05)*, Online *(ACE05)*, Ancient *(AncientTimes)*, Clinical *(i2b2 2012)*, Pubmed *(Time4SCI)*, SMS *(Time4SMS)*, Wiki *(WikiWars)*

Take a look at [src/dataloader.py](src/dataloader.py) to see the expected directory structure and adjust the path variables accordingly. 
The code is expecting a local installation of the BERT models specified with `--embed_base_path`. These models can be downloaded and stored from huggingface fur further local processing. 

### Training on source domain(s)

    python main_train.py -t NER -d News -o model

### Finetuning on target domain
The `--swap_heads` option has to be used for when a different label set was used during pretraining, i.e., in this paper for NER transfer experiments. 

    python main_train.py -t NER -d News -p model/best-model.pt --swap_heads -o model-2
	
## Model results and source prediction
We list our model results in the [results](results) directory for the different settings. These numbers can be used as a benchmark for new similarity measures and multi-source preditors. 
The implementation of our model similarity measures as well as the other baseline distances can be found in the [src/distances.py](src/distances.py) file. The code for all multi-source predictors is given in [src/distances.py](src/distances.py). The implemented evaluation metrics are provided in the code files as well. 

## License

Project-title is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

The software including its dependencies may be covered by third party rights, including patents. You should not execute this code unless you have obtained the appropriate rights, which the authors are not purporting to give.