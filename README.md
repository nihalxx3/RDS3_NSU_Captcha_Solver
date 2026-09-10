# RDS3 CAPTCHA Solver Machine Learning Model

A FastAPI-based backend service for solving 4-digit CAPTCHA images using PyTorch models, paired with a Tampermonkey userscript for automated browser login.

<div align="center">
  <img src="https://i.ibb.co.com/SXMzxDjn/image.png" alt="Project Showcase 1" />
  <br><br>
  <img src="https://i.ibb.co.com/rG0SY3yX/9792.png" alt="Project Showcase 2" />
  <p>9792 showing that it solved as 9792 captcha</p>
</div>

## Authors:

@nihalxx3 & @Tanvir-Chowdhury

## DISCLAIMER

This project was developed strictly for educational purposes. The authors have used this project for over 2 years on the RDS3 North South University website solely for the ease of login. No automation has been conducted using this project to date.

We strictly advise everyone to use this project exclusively for their personal use. Please avoid any kind of automation or bulk processing which may violate terms of service and result in institutional disciplinary actions.

All the code has been prepared with the help of multiple LLM models out there.

Warning: Users are strongly encouraged to read through the complete source code at least once before using it. The authors are not responsible or liable for any misuse, damage, or repercussions arising from the use of this project code.

## System Requirements

- Python 3.8 or higher
- Tampermonkey Browser Extension (Chrome/Firefox/Edge)

## Phase 1: Backend Setup

### 1. Configure the Environment

Navigate to the directory where your `main.py` is located.

### 2. Install Dependencies

Install the required Python packages using pip:

`pip install -r requirements.txt`

### 3. Start the Server

Run the FastAPI application. It is configured to run on port 3333.

`python main.py`

The API will be available at: http://localhost:3333
You can view the interactive API documentation at: http://localhost:3333/docs

## Phase 2: Browser Extension Setup

### 1. Install Tampermonkey

Install the Tampermonkey extension for your web browser if you have not already.

### 2. Add the Script

1. Click the Tampermonkey extension icon in your browser and select "Create a new script".
2. Delete any default template code provided.
3. Paste the provided JavaScript userscript (captcha_solver.js) into the editor. 

## Phase 3: Usage

1. Ensure the Python backend server (`python main.py`) is actively running in a terminal window.
2. Navigate to the RDS login page
3. The Tampermonkey script will automatically:
   - Detect the login form.
   - Auto-fill your configured password.
   - Extract the CAPTCHA image and send it to your local backend (http://localhost:3333).
   - Fill in the solved CAPTCHA code.
   - Submit the form to log you in.
