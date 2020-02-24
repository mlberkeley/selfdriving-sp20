# selfdriving-sp20

Self Driving Car Decal taught by Machine Learning @ Berkeley, Spring 2020 at UC Berkeley.

Questions? Post on Piazza

Week 1 Introduction:

Lecture Slides: https://docs.google.com/presentation/d/1m_08cHpmsF8a-8SVDKsQ8kDhvl-cWh3gSNx28wQF2Fs/edit?usp=sharing

Introduce idea of Self Driving Cars along with class discussion regarding preconceptions.  Lay out roadmap for course and get the simulator environment setup.
* Look at:
  * controller_api.txt
* Work with:
  * simulator.py
  * car_iface/controller.pyc
  * environment.yaml


Week 2 System ID:

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
