# Data

### Topics Covered:
- **Role of Data in AI**
- **Types of Data**
- **Data Quality**
- **Big Data**

### Objectives:
1. Understand the critical role of data in AI.
2. Identify and categorize different types of data used in AI, including structured, unstructured, and specific forms like record data, graph-based data, and ordered data.
3. Recognize the importance of data quality and its impact on AI models.
4. Gain an introductory understanding of Big Data and its applications in AI.

---

## 1. The Role of Data in AI

Data is the foundation of AI. AI systems learn from data, identify patterns, and make decisions based on the information they receive. Without data, AI models cannot be trained, validated, or deployed effectively.

### Key Points:
- **Training AI Models:** AI algorithms rely on large datasets to learn and make predictions. The quality and quantity of the data directly influence the performance of the model.
- **Data-Driven Decision Making:** AI systems use data to make decisions, whether it’s predicting trends, classifying objects, or generating recommendations.
- **Feedback Loop:** AI systems often improve over time as they process more data, continuously learning and refining their predictions.

---

## 2. Types of Data

Understanding the types of data used in AI is essential for choosing the right algorithms and preprocessing techniques. We can divided data at least from two different aspects:

## Data types by their structure:
### 2.1 Structured Data
- **Definition:** Data that is organized in a predefined manner, often in tabular form (e.g., spreadsheets, SQL databases).
  - **Examples:** Customer records, financial transactions, sensor readings.

### 2.2 Unstructured Data
- **Definition:** Data that does not have a predefined structure and is often more complex to analyze.
  - **Examples:** Text documents, images, audio, and video files.

### 2.3 Semi-Structured Data
- **Definition:** Data that does not conform to a rigid structure but has some organizational properties (e.g., JSON, XML files).
  - **Examples:** Emails, logs, and HTML documents.

## Data types by their nature:
### 2.4 Record Data
- **Definition:** Record data is data where each record (or row) represents an individual entity or transaction.
- **Transaction or Market-Based Data:** 
  - **Examples:** Purchase histories in e-commerce, financial transactions in banks.
  - **Use Case:** Analyzing customer behavior, detecting fraudulent activities.
- **Traditional Record Data:**
  - **Examples:** Employee databases, medical records.
  - **Use Case:** Organizing and retrieving information systematically.

### 2.5 Graph-Based Data
- **Data with Relationships Among Objects:**
  - **Definition:** Graph-based data includes entities (nodes) connected by relationships (edges).
  - **Examples:** Social networks, citation networks.
  - **Use Case:** Analyzing social connections, identifying influential nodes in a network.
- **Data with Objects That Are Graphs:**
  - **Definition:** Complex data structures where individual objects themselves are graphs.
  - **Examples:** Chemical compounds (molecules as graphs), network topologies.
  - **Use Case:** Predicting molecular properties, optimizing network designs.

### 2.6 Ordered Data
- **Sequential Data:**
  - **Definition:** Data that is ordered by nature, often related to time or sequences of events.
  - **Examples:** User clickstreams, gene sequences.
  - **Use Case:** Predicting the next user action, understanding biological sequences.
- **Sequence Data:**
  - **Definition:** A type of ordered data where the order of elements is significant.
  - **Examples:** DNA sequences, text sequences in natural language processing.
  - **Use Case:** Sequence alignment, language modeling.
- **Time Series Data:**
  - **Definition:** Data points collected or recorded at specific time intervals.
  - **Examples:** Stock prices, weather data.
  - **Use Case:** Forecasting, anomaly detection.
- **Spatial Data:**
  - **Definition:** Data that represents objects defined in a geometric space.
  - **Examples:** Geographic information systems (GIS), satellite imagery.
  - **Use Case:** Mapping, spatial analysis.

---

## 3. Data Quality

Data quality is crucial for the success of AI models. Poor-quality data can lead to inaccurate predictions, biased models, and unreliable outcomes.

### Key Aspects of Data Quality:
- **Accuracy:** Data should be correct and free from errors.
- **Completeness:** All necessary data should be present, with no missing values.
- **Consistency:** Data should be consistent across different datasets and sources.
- **Timeliness:** Data should be up-to-date and relevant to the current context.
- **Validity:** Data should conform to defined formats and standards.
_[More detail + Python notebooks](https://github.com/Naviden/Data-Quality-Issues)_

### Impact of Poor Data Quality:
- **Bias in Models:** Inaccurate or incomplete data can introduce bias, leading to unfair or incorrect predictions.
- **Overfitting/Underfitting:** Poor-quality data can cause models to overfit or underfit, reducing their generalizability.
- **Inefficiencies:** Models trained on low-quality data may require more computational resources and time to achieve acceptable performance.

---

## 4. Introduction to Big Data

Big Data refers to extremely large datasets that are difficult to manage, process, and analyze using traditional data-processing tools. Big Data is characterized by the "3 Vs": Volume, Velocity, and Variety.

### Characteristics of Big Data:

- **Volume**:  
  - **Definition**: The sheer amount of data generated every day.  
  - **Examples**: Social media posts, sensor data, transaction records.  
  - **Use Case**: Storage and processing systems like distributed file systems (e.g., Hadoop HDFS).  

- **Velocity**:  
  - **Definition**: The speed at which data is generated, collected, and processed.  
  - **Examples**: Real-time data streams from IoT devices, stock market feeds.  
  - **Use Case**: Real-time analytics, fraud detection, and dynamic pricing systems.  

- **Variety**:  
  - **Definition**: The diversity of data formats and types.  
  - **Examples**: Structured data (databases), unstructured data (text, images), and semi-structured data (JSON, XML).  
  - **Use Case**: Integrating and analyzing data from multiple sources, such as combining text, images, and videos for multimedia analytics.  

- **Veracity**:  
  - **Definition**: The uncertainty or reliability of data.  
  - **Examples**: Inconsistent, noisy, or incomplete data from social media or user-generated content.  
  - **Use Case**: Ensuring data quality for decision-making processes, such as customer feedback analysis.  

- **Value**:  
  - **Definition**: The usefulness or relevance of data in generating insights and driving decisions.  
  - **Examples**: Business intelligence derived from sales data or customer segmentation.  
  - **Use Case**: Monetizing data through predictive analytics, personalized marketing, or operational optimizations.  
### Big Data in AI:
- **Scalability:** AI models must be scalable to handle and learn from vast amounts of data.
- **Advanced Analytics:** Big Data allows for more complex and comprehensive analysis, leading to better insights and predictions.
- **Applications:** Big Data is used in various AI applications such as predictive analytics, recommendation systems, and natural language processing.

---

### Recommended Reading:
- **["Introduction to Data Mining" by Pang-Ning Tan, Michael Steinbach, and Vipin Kumar](https://www.ceom.ou.edu/media/docs/upload/Pang-Ning_Tan_Michael_Steinbach_Vipin_Kumar_-_Introduction_to_Data_Mining-Pe_NRDK4fi.pdf)**