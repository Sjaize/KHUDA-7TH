# Korean Grammar Correction Generator

This project uses CLOVA X API to generate explanations for Korean grammar corrections.

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set up your API key in the code or use environment variables.

3. Prepare your input CSV file with columns:
- Wrong_S: Incorrect expression
- Right_S: Correct expression
- Reason_0_2: Error type code

## Usage
Run the main script:
```bash
python main.py
```

The script will process your input CSV and generate explanations for each grammar correction.
