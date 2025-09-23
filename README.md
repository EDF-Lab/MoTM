# TimeFlow Imputation  

## 1. Overview  

This repository provides
 1. An implementation of **TimeFlow**, a deep learning model for time series, jointly developed by **EDF R&D** and **Sorbonne Université**;
 2. An implementation of **MoTM**, a step towards a foundation model for time series imputation. 

In a nutshell, TimeFlow is a **time-continuous deep learning model** designed for time series **imputation** and **forecasting** (this repository focuses on the imputation framework). It leverages **implicit neural representations (INRs)**, **auto-decoding**, and **meta-learning** to model and infer missing point in time series datasets. TimeFlow addresses common real-world challenges such as **irregular sampling**, **missing data**, and **unaligned multi-sensor measurements**.

**MoTM** combines a basis of INRs, each trained independently on a distinct family of time series, with a ridge regressor that adapts to the observed context
at inference.

For further details, please check:  
📖 [Concepts](00-concepts.md)  
📄 [TimeFlow Paper](https://arxiv.org/pdf/2306.05880)  
📄 [MoTM Paper](https://arxiv.org/pdf/2507.13207)  

### Authors  

- **Code Contributors:** Etienne Le Naour, Tahar Nabil
- **TimeFlow Authors:** Etienne Le Naour, Louis Serrano, Léon Migus, Yuan Yin, Ghislain Agoua, Nicolas Baskiotis, Patrick Gallinari, Vincent Guigue.  

---

## 2. Project Structure  

...

---

## 3. Running the Project  

### Environment Setup  

All dependencies are listed in [`pyproject.toml`](pyproject.toml).  
The scripts in [`scripts/`](scripts/) are designed for `uv`, but can be adapted for other environments like `conda`.  

### Instructions  

...

### Key Information  


---

## 4. Contact  

💬 Have questions, found a bug, or need specific features?  
Feel free to reach out!  

📧 **Contact:**  
- [Etienne Le Naour](mailto:etienne.le-naour@edf.fr?subject=TimeFlow%20Imputation)  
- [Tahar Nabil](mailto:tahar.nabil@edf.fr?subject=TimeFlow%20Imputation)  

If you find this repository useful, **please consider citing TimeFlow / MoTM and starring the repo**! ⭐  