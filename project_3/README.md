# 🎓 Student Portfolio — AI & Python Projects

> A responsive, dark-themed personal portfolio website built with vanilla HTML, CSS & JavaScript — showcasing AI/ML projects completed during a hands-on workshop.

![Built with HTML CSS JS](https://img.shields.io/badge/Built%20with-HTML%20%7C%20CSS%20%7C%20JS-7C4DFF?style=flat-square)
![Status](https://img.shields.io/badge/Status-Ready%20for%20GitHub-00E676?style=flat-square)
![Responsive](https://img.shields.io/badge/Responsive-Yes-00E5FF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-white?style=flat-square)

---

## 📸 Preview

> _Replace this section with a screenshot once deployed:_
> `![Portfolio Preview](./preview.png)`

---

## 🚀 Live Demo

> [🌐 View Portfolio →](https://your-username.github.io/your-repo-name)  
> _(Update this link after deploying to GitHub Pages)_

---

## ✨ Features

- **Animated particle canvas background** — floating dots with gradient connecting lines
- **Smooth scroll reveal** — sections animate in as you scroll using IntersectionObserver
- **Typing effect** — rotating taglines in the hero section
- **Scroll progress bar** — fixed gradient bar at the top of the page
- **Responsive design** — mobile drawer menu, fluid grid layouts
- **Glassmorphism UI** — frosted glass cards with gradient borders
- **Animated gradient blobs** — soft ambient lighting in the background
- **Easy personalization** — update one `LINKS` object in JavaScript to set all URLs
- **URL query param support** — `?name=YourName&github=<url>` for quick customization

---

## 🗂️ Sections

| Section | Description |
|---|---|
| **Hero** | Introduction, typing animation, and quick stats |
| **About** | Who I am, what I'm learning, mentorship |
| **Skills & Tools** | Python, Pandas, Matplotlib, EDA, scikit-learn |
| **Featured Projects** | EDA Dashboard (Titanic) & House Price Prediction |
| **Workshops & Learning** | Ardent AI/ML Workshop + Code_ScholarEU |
| **Contact** | Email, LinkedIn, Instagram links |

---

## 🧪 Projects Showcased

### 📊 Project 1 — EDA Dashboard (Titanic Dataset)
Exploratory Data Analysis on the Titanic dataset.
- Loaded and explored data using **Pandas**
- Handled missing values with mean/mode imputation
- Created visualizations: survival count, gender vs survival, age distribution
- Summarized key insights for presentation

**Stack:** `Python` `Pandas` `Matplotlib` `Google Colab`

---

### 🏠 Project 2 — House Price Prediction
First supervised ML model using Linear Regression.
- Train-test split to evaluate on unseen data
- Trained with **scikit-learn** `LinearRegression`
- Evaluated using **RMSE** and **R² score**
- Visualized Actual vs Predicted and residual errors

**Stack:** `Python` `scikit-learn` `Matplotlib` `Google Colab`

---

## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| HTML5 | Structure and semantics |
| CSS3 | Custom properties, animations, glassmorphism, responsive grid |
| Vanilla JavaScript | Typing effect, particle canvas, scroll reveal, mobile menu |
| Font Awesome 6 | Icons throughout the UI |
| Google Colab | ML project development |

---

## ⚙️ How to Personalize

Open `index.html` and find the `LINKS` object near the bottom of the `<script>` tag:

```javascript
const LINKS = {
  githubProfile:    "https://github.com/your-username",
  linkedin:         "https://linkedin.com/in/your-profile",
  project1Repo:     "https://github.com/your-username/repo-name",
  project2Repo:     "https://github.com/your-username/repo-name",
  project1Notebook: "https://colab.research.google.com/...",
  project2Notebook: "https://colab.research.google.com/...",
  email:            "your@email.com"
};
```

Also update the `Student Name` text in the `<h1>` tag and any other hardcoded text in the HTML.

---

## 📁 File Structure

```
📦 portfolio/
 ┗ 📄 index.html      ← Single-file portfolio (HTML + CSS + JS)
 ┗ 📄 README.md       ← This file
 ┗ 🖼️ preview.png     ← (Optional) Screenshot for README
```

---

## 🚢 Deployment (GitHub Pages)

1. Push this repository to GitHub
2. Go to **Settings → Pages**
3. Set source to `main` branch, root folder
4. Your site will be live at `https://your-username.github.io/repo-name`

---

## 🎓 About

**Student:** B.Sc (Computer Science), 2nd Year — 4th Semester  
**Institution:** Haldia Institute of Management, Haldia, India  
**Mentor:** SK Sahil — AI Developer & Tutor ([Code_ScholarEU](https://www.instagram.com/code_scholar_eu/))  
**Workshop:** Ardent — AI & Machine Learning (3-day hands-on, Google Colab)

---

## 📬 Contact

- 📧 Email: `student@email.com` _(replace with your email)_
- 💼 LinkedIn: [your-profile](https://linkedin.com/in/your-profile)
- 📸 Instagram: [@code_scholar_eu](https://www.instagram.com/code_scholar_eu/)
- 🐙 GitHub: [your-username](https://github.com/your-username)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">Built with 💜 using HTML, CSS & JavaScript · Ready for GitHub Pages</p>
