# Smart-Glass-Medical-Assistant
Employing glass AI optical character recognition (OCR system, to capture images of a PDF document in the user’s view. Then convert the images to text into a digital format. After this information is analyzed, various functions to be completed: summarizing the documentation, location of specific information, and converting text to audio.

## System Architecture
- Overview
The recommended architectural pattern is Layered Architecture. This structure is ideal for systems that
require clear separation of concerns, which is critical for maintainability and accommodating future
student teams.
Reasoning

- Concerns are Separated: The ability for each layer to be developed and updated independently.
Example: If the "Business Logic" changes (i.e. summarization algorithms, RAG checks) then the
"Presentation" (i.e. Smart Glasses Interface) Layers are not affected.

- Maintainability: Future teams will not have trouble isolating the source of any problem or where
a Feature needs to be added.

- Testability: The different Layers are able to Test Layers individually, particularly the Business
Logic and Data Access Layers.

- Security: Security Verification can occur at both the Application and Data Access Layers, prior
to being made available to the Presentation Layer.


