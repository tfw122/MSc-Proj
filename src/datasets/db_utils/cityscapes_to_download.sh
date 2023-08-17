
# Change $USERNAME and $PASSWORD to your credentials
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=tiafw122&password=sushi.c0M&submit=Login' https://www.cityscapes-dataset.com/login/

# Install left images
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=14

# Install right images
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=15

unzip ./leftImg8bit_sequence_trainvaltest.zip
unzip ./rightImg8bit_sequence_trainvaltest.zip
