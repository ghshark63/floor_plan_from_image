# Floor plan from image
Our project is a tool that generates 2d floor plan from the given list of images. 
It will combine multiple instruments to create a pipeline to processed source images become 2d schema.
Users will be able to specify how rooms are connected with each other.
These and other product capabilities aren't fixed for now and can be changed in future along with our domain investigations.

To transform simple photos to abstract point clouds We are going to test:

1) combination of Structure from Motion + Multi-View Stereo 

2) Master SLAM


Also we plan to train/fine-tune model to detect type of furniture, then map it to 2d schema.

Project's target audience are the people that want to sell real estate but don't have its up-to-date detailed floor plan,
so they can just create it from simple rooms' photos. 
Its importance directly follows from the problem - improve acquirer's understanding of the layout of expensive real estate.
