# Model Deployment

## Model Deployment overview

Model Deployment: Bring learned models to different application environments. There are some considerations:

- Application environment may bring restrictions(model size, no-python)

- Leverage local hardware acceleration(mobile GPUs, accelerated CPU instructions, NPUs)

- Integration with the applications(data preprocessing, post processing)

Inference engine internals:

- Many inference engines are structed as computational graph interpreters

- Allocate memories for intermediate activations, traverse the graph and execute each of the operators

- Usually only support a limited set of operators and programming models(e.g. dynamism)


## Machine learning compilation 

Limitation of library driven inference engine deployments

- Need to build specialized libraries for each hardward backend

- A lot of engineering efforts to optimization 

The steps of ML compilation flow:

- Model -> Intermediate representation(IR)

- Intermediate representation -> High-level transformations 

- High level transformations -> lowering to loop IR 

- loop IR -> Low-level transformations

- Code generation and execution 


Elements of an automated ML compiler

- Program representation

    -  Representation the program/optimization of interest, (e.g. dense tensor linear algebra, data structures)
    
- Build search space througn a set of transformations 

    - Cover common optimizations

    - Find ways for domain experts to provide input 

- Effective Search (Still an open research area!!) 
    - Cost models. transferability
