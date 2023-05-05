DATA_FOLDER="/home/layout/layout-parser/data"
if [ ! -d $DATA_FOLDER ]
then
	echo "Data folder does not exists. Create and rerun."
	exit -1
fi

docker build -f containers/Dockerfile -t parser:dbnet .
if [ "$?" != "0" ]
then
	echo "Failed to build the container"
	exit -1
fi

docker run -it --gpus all -v "${DATA_FOLDER}":/data  parser:dbnet containers/commands.sh
if [ "$?" != "0" ]
then
	echo "Failed to setup the container for inference"
	exit -1
fi

# Example run command for inference
# IMAGES_FOLDER="/home/layout/layout-parser/images"
# docker run -it --gpus all -v "$DATA_FOLDER":/data  -v "$IMAGES_FOLDER":/images parser:dbnet containers/run-dbnet /data/dbnet /images
