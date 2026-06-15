\# AR Virtual Try-On



An Augmented Reality (AR) Virtual Try-On application that allows users to visualize clothing items on themselves in real time using a webcam.



\## Overview



This project uses computer vision and pose estimation techniques to overlay virtual clothing on a user's body. The application captures live video from a webcam, detects body landmarks using MediaPipe, and dynamically positions clothing images to create an interactive virtual try-on experience.



\## Features



\* Real-time webcam integration

\* Virtual clothing overlay

\* Human pose detection using MediaPipe

\* Interactive AR experience

\* Web-based interface using Flask

\* Image processing with OpenCV



\## Technologies Used



\* Python

\* Flask

\* OpenCV

\* MediaPipe

\* NumPy

\* Pillow

\* HTML

\* CSS



\## Project Structure



```text

AR-Virtual-TRY-ON/

│

├── Resources/

├── static/

├── templates/

├── app.py

├── requirements.txt

└── README.md

```



\## Installation



1\. Clone the repository:



```bash

git clone https://github.com/hyni19/AR-Virtual-TRY-ON.git

```



2\. Navigate to the project directory:



```bash

cd AR-Virtual-TRY-ON

```



3\. Create a virtual environment:



```bash

python -m venv venv

```



4\. Activate the virtual environment:



Windows:



```bash

venv\\Scripts\\activate

```



5\. Install dependencies:



```bash

pip install -r requirements.txt

```



\## Running the Application



```bash

python app.py

```



Open your browser and visit:



```text

http://127.0.0.1:5000

```



\## Future Enhancements



\* Multiple clothing categories

\* Improved garment fitting

\* User profile customization

\* Mobile compatibility

\* Enhanced AR accuracy



\## Author



Hyni Jerusha Raj Dosapati

B.Tech – Artificial Intelligence and Machine Learning



