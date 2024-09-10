Thin Layer Chromatography (TLC) Plate RF Measurement App
This is a Python application built using Kivy and OpenCV to assist in measuring the Retention Factor (Rf) on a Thin Layer Chromatography (TLC) plate. The app allows users to capture a photo of the TLC plate, automatically crop the image if the contrast is sufficient, and perform Rf calculations based on the position of the spots on the plate.

Features
Automatic Cropping: Utilizing OpenCV, the app automatically crops the image of the TLC plate if the contrast between the plate and its surroundings is adequate.
Retention Factor (Rf) Calculation: Measure the Rf of the chromatographic spots based on the positions and the distance traveled by the solvent front.
Solvent Management: Add and manage solvent information used during chromatography.
Save Results: Save the Rf results and solvent information into a file for further analysis or record-keeping.
Installation
Clone the repository:
bash
Copiar código
git clone https://github.com/yourusername/tlc-rf-measurement.git
Install the dependencies:
bash
Copiar código
pip install -r requirements.txt
Usage
Run the application:
bash
Copiar código
python main.py
Capture or upload a photo of the TLC plate.
The app will automatically crop the image if the contrast is sufficient.
Input the solvent front and spot distances, and the app will calculate the Rf value.
Save the results as needed.
Requirements
Python 3.x
Kivy
OpenCV
NumPy
License
This project is licensed under the MIT License. See the LICENSE file for details.



The link is a compiled file that works on a PC with Windows. https://ouo.io/Yk647Y
