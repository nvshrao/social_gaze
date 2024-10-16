## Data

Please find the data [here](https://drive.google.com/drive/folders/1Dx7girZJOByU7GtoCvrZKCknK5b8U0V2?usp=sharing).  
You can paste the file `evaluate_rel_with_score.csv` into the `./data/` directory.

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
