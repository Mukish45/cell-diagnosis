# Cell Image Disease Classification (EDA)

To run the code, you need to install the following packages:

- tensorflow
- cv2
- os
- shutil
- matplotlib
- zipfile

or

you can use ```environment.yml``` file to create a conda environment.

    conda env create -f environment.yml

for Pip users:

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

## Dataset

The dataset contains 27,558 cell images with equal instances of parasitized and uninfected cells. There will be 13780 parasitized and 13780 uninfected cell samples. All these samples are present in the Google drive. Mount the Drive with Colab Notebook to access the data.

## Exploratory Data Analysis(EDA):
- Visualization of images
- Class imbalance
- Image size variability

Load any three images using the following code:

    img1 = cv2.imread('/content/drive/MyDrive/Cell-Img-Data/dataset/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png')

    img2 = cv2.imread('/content/drive/MyDrive/Cell-Img-Data/dataset/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_166.png')

    img3 = cv2.imread('/content/drive/MyDrive/Cell-Img-Data/dataset/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_171.png')

To visualize the images:

    fig = plt.figure(figsize=(18, 6))
    axarr = fig.subplots(1, 3)

    image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    axarr[0].imshow(image)

    image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    axarr[1].imshow(image)

    image = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    axarr[2].imshow(image)

    plt.show()
    
 
 To check the number of samples in each folder:
 
    source = '/content/drive/MyDrive/Cell-Img-Data/dataset/Parasitized'
    dir1_lst = os.listdir(source)
    print(len(dir1_lst))
    
 To check image size:
 
    print(img1.shape)
    print(img2.shape)
    print(img3.shape)
    
## Data Splitting:
The dataset is split into three groups namely
- Training set (includes 75% of samples)
- Validation set (includes 15% of samples)
- Evaluation / Test set (includes 10% of samples)

Create separate folders for train, validation and evaluation which includes Parasitized and Uninfected cells:
    
    os.mkdir("train")
    
    os.mkdir("valid")
    
    os.mkdir("eval")
    
    frame_dir = "/content/drive/MyDrive/Cell-Img-Data/dataset/Parasitized"
    frame_dir_lst = os.listdir(frame_dir)
    frame_dir_lst.sort()

    output_dir1 = "/content/train/parasitized"

    if not os.path.exists(output_dir1):
      os.mkdir(output_dir1)

    for image_nm in frame_dir_lst[:10335]:
      img = cv2.imread(os.path.join(frame_dir,image_nm))
      cv2.imwrite(os.path.join(output_dir1,image_nm),img)


    output_dir2 = "/content/valid/parasitized"

    if not os.path.exists(output_dir2):
      os.mkdir(output_dir2)

    for image_nm in frame_dir_lst[10335:12402]:
      img = cv2.imread(os.path.join(frame_dir,image_nm))
      cv2.imwrite(os.path.join(output_dir2,image_nm),img)



    output_dir3 = "/content/eval/parasitized"

    if not os.path.exists(output_dir3):
      os.mkdir(output_dir3)

    for image_nm in frame_dir_lst[12402:]:
      img = cv2.imread(os.path.join(frame_dir,image_nm))
      cv2.imwrite(os.path.join(output_dir3,image_nm),img)
      
## Image Data Generator
After splitting the data into three folders, use ImageDataGenerator to manage the data. It can be used to resize image, 


## Model Architecture
- Convolution layer with 9x9 kernel
- Convolution layer with 6x6 kernel
- Fully-connected layer with the appropriate number of neurons
- Fully-connected layer with the appropriate number of neurons and a dropout of 50% probability
- Fully-connected layer with the appropriate number of neurons

        initializer = tf.keras.initializers.GlorotUniform()
        model = Sequential()
        model.add(layers.Conv2D(input_shape=(150, 150, 3), filters=64,kernel_size=(9,9), padding="same",activation='relu'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', padding='same', name='block1_conv1'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu', kernel_initializer=initializer))
        model.add(Dropout(0.5))
        model.add(layers.Dense(128, activation='relu', kernel_initializer=initializer))

        layer = tf.keras.layers.Dense(1, kernel_initializer=initializer, activation='sigmoid')#The Glorot uniform initializer, also called Xavier uniform initializer.
        model.add(layer)

# Synopsys

## Title of the Project : Cell Image Disease Classification

## Aim of the Project : To use the cell images and identify parasitized and uninfected cells.
