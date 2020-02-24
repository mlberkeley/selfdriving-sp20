# selfdriving-sp20

**Self Driving Cars Decal** taught by Machine Learning @ Berkeley, Spring 2020 at UC Berkeley.

**Quick Links**:
* **Anonymous Feedback**: https://forms.gle/wCKxfH3sT87RzQey7 
* **Weekly Checkoff**: https://forms.gle/9DfNj87bd9cFiSKh9
* **Groups**: https://docs.google.com/spreadsheets/d/110xZ6lQH14uPunvVVar2yfUGFXM4I8qXt4PxjiCKqVA/edit?usp=sharing
* **Piazza**: https://piazza.com/class/k6ip3fsmllu4zw
* **Anaconda Commands**: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


**Week 1 Introduction**:

Lecture Slides: https://docs.google.com/presentation/d/1m_08cHpmsF8a-8SVDKsQ8kDhvl-cWh3gSNx28wQF2Fs/edit?usp=sharing

Introduce idea of Self Driving Cars along with class discussion regarding preconceptions.  Lay out roadmap for course and get the simulator environment setup.
* Look at:
  * controller_api.txt
* Work with:
  * simulator.py
  * car_iface/controller.pyc
  * environment.yaml


**Week 2 System ID**:

Lecture Slides: https://docs.google.com/presentation/d/1ONr7fAf8cXZyqYt2meP5cXuFr58mxJz_LrFJ8KUhbAA/edit?usp=sharing

Build towards controlling the car to perform specific tasks by understanding how our controls (pedals, gears, steering) actually affect the cars state.  Approach Linear Regression from a gradient descent perspective, and learn the weights of the underlying car interface.

* Look at:
  * demos/week2/visual_simulation.py
  * demos/week2/housing_demo.py
* Work with:
  * simulator.py
  * car_iface/controller.pyc
* Write in:
  * car_iface/controller_model.py
  * hw/sysid/hw2_system_id.ipynb


**Week 3 Braking Distance**:

Lecture Slides: https://docs.google.com/presentation/d/1JEYCW1_ATtSKr7hCHCGafr5Gwy7krpcwIQVo6-I6sTw/edit?usp=sharing

Develop a model that based on how fast the car is moving adaptively stops precisely at customizable target locations.  Build on Linear Regression understanding, to learn about Fully Connected Neural Networks.  Use FCNs to model nonlinear internal car dynamics and for the adaptive braking algorithm.

* Look at:
  * utils/nn.py
  * demos/week3/Nonlinear_SystemID.ipynb
* Work with:
  * simulator.py
  * car_iface/controller.pyc
  * braking_distance/keypoints.py
* Write in:
  * car_iface/controller_model.py
  * hw/bd/hw3_braking_distance.ipynb
  * braking_distance/bd_api.py
  
