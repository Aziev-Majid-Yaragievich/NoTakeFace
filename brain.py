if __name__ == "__main__":
	from fastai.vision import *
	import numpy as np

	classes = ['face','hand-face']

	path = 'C:/Users/progr/Desktop/NoTakeFace'


	np.random.seed(42)
	data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
	    ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

	learn = create_cnn(data, models.resnet34, metrics=error_rate)


	learn = learn.fit_one_cycle(5)

	learn.save('stage-1')
	learn.export()

	img = open_image('C:/Users/progr/Desktop/NoTakeFace/photo.jpg')

	pred_class,pred_idx,outputs = learn.predict(img)

	if pred_class == 'hand-face':
		print('Вы трогаете своё лицо! Лучше перестать это делать!')

	elif pred_class == 'face':
		print('Вы вроде бы не трогаете своё лицо! Продолжайте в том же духе!')

	else:
		print("Хммм... Вы вроде бы не удалили папку 'Models' и файл 'export' из проекта! Попрошу вас это сделать!")