# SocialGaze

This repository contains the code and data for our paper:  
**[SocialGaze: Improving the Integration of Human Social Norms in Large Language Models](https://arxiv.org/abs/2410.08698v1)**.

## Data

Please find the data [here](https://drive.google.com/drive/folders/1Dx7girZJOByU7GtoCvrZKCknK5b8U0V2?usp=sharing).  
You can paste the file `evaluate_rel_with_score.csv` into the `./data/` directory.

Please use the data with care. We have retained gender and age information in the posts, as these are important for understanding social acceptance in models. No names or other personal identifying information were found in the dataset.

## Usage

## Usage

For Vanilla prompting:  
```bash
python vllm_aita.py phi-3-med
```

For SocialGaze prompting:  
```bash
python vllm_aita_delib.py phi-3-med
```
The generations will be saved into the `./results/` directory.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@misc{vijjini2024socialgazeimprovingintegrationhuman,
  title={SocialGaze: Improving the Integration of Human Social Norms in Large Language Models}, 
  author={Anvesh Rao Vijjini and Rakesh R. Menon and Jiayi Fu and Shashank Srivastava and Snigdha Chaturvedi},
  year={2024},
  eprint={2410.08698},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.08698},
}
