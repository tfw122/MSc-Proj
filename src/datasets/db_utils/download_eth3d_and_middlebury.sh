mkdir /middlebury -p
cd /middlebury/
wget https://www.dropbox.com/s/fn8siy5muak3of3/official_train.txt -P MiddEval3/
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip
unzip MiddEval3-data-Q.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip
unzip MiddEval3-GT0-Q.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip
unzip MiddEval3-data-H.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip
unzip MiddEval3-GT0-H.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip
unzip MiddEval3-data-F.zip
wget https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-F.zip
unzip MiddEval3-GT0-F.zip
rm *.zip
cd ../

mkdir /eth3d -p
cd /eth3d/
mkdir ./two_view_test
cd ./two_view_test
wget https://www.eth3d.net/data/two_view_test.7z
echo "Unzipping two_view_test.7z using p7zip (installed from environment.yaml)"
7za x two_view_test.7z
cd ../

mkdir ./two_view_training
cd ./two_view_training
wget https://www.eth3d.net/data/two_view_training.7z
echo "Unzipping two_view_training.7z using p7zip (installed from environment.yaml)"
7za x two_view_training.7z
cd ../

mkdir ./two_view_training_gt
cd ./two_view_training_gt
wget https://www.eth3d.net/data/two_view_training_gt.7z
echo "Unzipping two_view_training_gt.7z using p7zip (installed from environment.yaml)"
7za x two_view_training_gt.7z
cd ../