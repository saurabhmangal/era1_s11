**This is the submission for assigment number 11 of ERA V1 course.**<br> 

**Problem Statement**<br> 
The Task given was to use CIFAR 10 data and uses resnet model and monitor the accuracy of the model for 20 EPOCHS. Further, show 10 GRADCAM images. The code should be wriiten in a modular way.  

**File Structure**<br> 
-resnet.py           - has the resnet model copied from the github library suggested in the assignment<br>
-era_s11_cifar.ipynb  - the main .ipynb file<br> 
-Colab_notebook.ipynb - Google Colab file to executed<br> 
-main.py              - main file in .py mode
-plots.py             - contains function to plot
-utils.py             - contains different functions which are 
                        Cifar10SearchDataset, 
                        create_train_data_loader, 
                        create_test_data_loader,
                        train_transforms,
                        test_transform, 
                        imshow, 
                        display_mis_images, 
                        learning_r_finder, 
                        OneCycleLR_policy
-calc_loss_accuracy.py - function to train and test loss and accuracy while model training
-images:<br> 
  -Accuracy & Loss.jpg        -- Plot of train and test accuracy and loss with respect to epochs<br> 
  -miss_classified_image.jpg  -- sample mis classified images. <br> 
  -test_dataset.jpg           -- sample test dataset<br> 
  -train_dataset.jpg          -- sample train dataset after tranformation<br> 

The tranformation performed as as follows:<br> 

    def train_transforms(means,stds):
        transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                ToTensorV2(),
            ]
        )

    def test_transforms(means,stds):
        transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )
        return transforms
        
Following are the sample images of train dataset:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/test_dataset.jpg" alt="alt text" width="600px">

Following are the sample imagese of the test dataset:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/test_dataset.jpg" alt="alt text" width="600px">


**Custom Resnet ARCHITECTURE**<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/model10.JPG" alt="alt text" width="600px">


**Last Epoch Results:**<br>
EPOCH: 23<br>
Loss=0.042819224298000336 LR =-1.5486702470463194e-06 Batch_id=48 Accuracy=98.64: 100% 49/49 [00:09<00:00,  5.15it/s]<br>
Test set: Average loss: 0.0002, Accuracy: 9239/10000 (92.39%)<br>

Following are the plot of train and test losses and accuracies:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/Accuracy%20%26%20Loss.jpg" alt="alt text" width="600px"><br> 

Some of the sample misclassified images are as follows:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/mis_classified_image.jpg" alt="alt text" width="600px"><br> 

Plot for One Cycle LR policy:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/oneLRcurve.png" alt="alt text" width="600px"><br> 

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
