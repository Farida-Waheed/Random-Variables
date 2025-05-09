# 🎲 Random Variables Simulator & Visualizer

A comprehensive Python-based project that demonstrates the theoretical background and practical implementation of **Discrete and Continuous Random Variables**. This project simulates, analyzes, and visualizes multiple probability distributions using custom functions and Matplotlib.

---

## 🏫 Project Information

* **University:** Benha University, Faculty of Engineering (Shoubra)  
* **Department:** Communications and Computer Engineering  
* **Course:** Engineering Mathematics (5)  

---

## 👩‍💻 Team Members

* Abdelrahman Salah El-dein Abdelaziz  
* Abdullah Mohamed Mohamed Galal  
* Farida Waheed Abd El Bary  
* Mohamed Ahmed Mohamed Hassan  
* Nour Hesham El Sayed  
* Omar Sami Mohamed Ahmed  
* Razan Ahmed Fawzy  

---

## 📌 Project Overview

This project offers a detailed exploration of:

- **Discrete Random Variables**: Bernoulli, Binomial, Geometric, Uniform, and Poisson distributions
- **Continuous Random Variables**: Uniform, Exponential, and Gaussian distributions
- Their **theoretical foundations**, **formulas**, **real-life applications**, and **Python-based simulations**

---

## 📊 Distributions Covered

### 📘 Discrete Random Variables

| Distribution | Description |
|--------------|-------------|
| **Bernoulli** | Binary success/failure model |
| **Binomial**  | Probability of k successes in n trials |
| **Geometric** | Trials until the first success |
| **Discrete Uniform** | Equal probability for each outcome |
| **Poisson**   | Number of events in a fixed interval |

### 📗 Continuous Random Variables

| Distribution | Description |
|--------------|-------------|
| **Uniform**  | Equal likelihood over an interval |
| **Exponential** | Time between events |
| **Gaussian (Normal)** | Bell-shaped curve used widely in statistics |

---

## 💡 System Features

* **Python Functions** for PMF, PDF, CDF, mean, and variance calculations
* **Histogram generation** to simulate real-world sampling
* **Graphical Visualizations** using Matplotlib
* **Custom logic** for sampling methods including inverse transform

---

## 🧾 How It Works

1. Each distribution includes:
   - PMF/PDF and CDF formulas
   - Mean and variance
   - Random variable generation
   - Histogram, PMF/PDF, and CDF plots

2. Plots are generated with proper labeling and annotations for:
   - Expected value (mean)
   - Variance

3. All functionality is implemented in `Random Variables.py`.

---

## ⚙️ Sample Functions

```python
bernoulli(probability=0.5, size=100)
binomial(n=10, p=0.3, size=1000)
geometric(probability=0.4, size=500)
uniform_discrete(a=1, b=6, size=1000)
poisson(lambda_=4, lower_bound=0, upper_bound=20, size=1000)
continous_uniform(lower_bound=0, upper_bound=10, npoints=100, size=1000)
exponential(lambda_=0.1, lower_bound=0, upper_bound=50, npoints=100, size=1000)
gaussian(mean=35, std_dev=3.5, size=1000)
````

---

## 📐 Algorithm Flow

1. Input parameters for the distribution
2. Generate random samples
3. Compute PMF/PDF and CDF
4. Plot:

   * Histogram
   * PMF/PDF line
   * CDF step function
5. Annotate with statistical measures

---

## 🧪 Testing Examples

| Function                  | Expected Behavior              | Passed |
| ------------------------- | ------------------------------ | ------ |
| `bernoulli(0.6, 100)`     | 60% 1s, 40% 0s + correct graph | ✅      |
| `binomial(10, 0.5, 1000)` | Distribution centered at 5     | ✅      |
| `gaussian(35, 3.5, 1000)` | Bell-shaped curve around µ=35  | ✅      |

---

## 💾 Files

* `Random Variables.py` – Python code with all implemented distributions
* `Random Variables.pdf` – Full theoretical documentation and real-life applications
* `Random Variables.pptx` – Presentation slides summarizing the topic

---

## 📷 Visual Outputs

> To be uploaded:

* Discrete Random Variables 
1. Bernoulli Distribution
   
   ![image](https://github.com/user-attachments/assets/ed9d6146-3e81-4c27-882c-c67d85f7621c)
2. Binomial Random Variable
   
   ![image](https://github.com/user-attachments/assets/b069a4c3-d1c6-415e-9867-d7d58facc155)
   ![image](https://github.com/user-attachments/assets/6db58b1c-48f4-492e-a8fb-eae694e889b4)
3. Geometric Random Variable
   
   ![image](https://github.com/user-attachments/assets/529dd808-9b0f-4174-9ce3-0faff5e4eb76)
4. Uniform Random Variable
   
   ![image](https://github.com/user-attachments/assets/b091c91d-c532-4bd2-a20e-5019761dc84c)
5. Poisson Random Variable
    
   ![image](https://github.com/user-attachments/assets/81c111eb-a1a3-4486-bd1f-7a7c1b903063)
* Continuous Random Variables 
1. Uniform Random Variable
   
   ![image](https://github.com/user-attachments/assets/6d538893-78eb-4aa6-ad27-a3bcdf45307d)
2. Exponential Distribution
   
   ![image](https://github.com/user-attachments/assets/f791612c-5877-4030-9341-39c46c5a5ca6)
   ![image](https://github.com/user-attachments/assets/37210260-60ab-49b4-868b-1204c3556d9e)

---

## 🛠 Tools Used

* **Python 3.11+**
* **Matplotlib** – Plotting and visualization
* **Math & Random Libraries** – Probability computations

