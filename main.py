import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import random
import pygame
pygame.init()

def draw_rect(pos, screen):
	rectsize = 5
	pygame.draw.rect(screen, 'black', (pos[0], pos[1], rectsize, rectsize))


def draw_img(screen):
	run = True
	drag = False

	screen.fill('white')

	while run:

		pygame.display.update()

		for event in pygame.event.get():
			if  (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN) or event.type == pygame.QUIT:
				image = pygame.Surface((280, 280))
				image.blit(screen, (0, 0))
				pygame.image.save(image, "drawn_img.png")
				pygame.quit()
				running = False
				return

			elif event.type == pygame.MOUSEBUTTONDOWN:
				drag = True
				pos = pygame.mouse.get_pos()
				screen.set_at(pos, 'black')
			
			elif event.type == pygame.MOUSEBUTTONUP:
				drag = False

			elif event.type == pygame.MOUSEMOTION:
				if drag:
					pos = pygame.mouse.get_pos()
					draw_rect(pos, screen)


def format_img(im):
	im = Image.open(im)

	resized = im.resize((28, 28))
	formatted = ImageOps.grayscale(resized)
	formatted = ImageOps.invert(formatted)

	im_arr = np.tile(np.array(formatted), (10000, 1, 1))

	return im_arr


def train_model(train_images, train_labels, test_images, test_labels):
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28, 28)),
		keras.layers.Dense(128, activation="relu"),
		keras.layers.Dense(10, activation="softmax")
	])
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	model.fit(train_images, train_labels, epochs=25)

	old_model = keras.models.load_model('model')

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	old_loss, old_acc = old_model.evaluate(test_images, test_labels)

	if test_acc > old_acc and test_loss < old_loss:
		print('Saving new model, ')
		model.save('model')


def show_prediction(prediction, test_images):
	# for _ in range(5):
	im_arr = format_img("drawn_img.png")
	image_index = random.randint(0, len(test_images))
	plt.grid(False)
	plt.imshow(im_arr[image_index], cmap=plt.cm.binary)
	plt.title("Prediction: " + str(np.argmax(prediction[image_index])))
	plt.show()


def main():
	data = keras.datasets.mnist
	(train_images, train_labels), (test_images, test_labels) = data.load_data()
	train_images = train_images / 255
	test_images = test_images / 255

	# train_model(train_images, train_labels, test_images, test_labels)
	model = keras.models.load_model('model')


	WIDTH, HEIGHT = 280, 280
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	draw_img(screen)

	prediction = model.predict(format_img("drawn_img.png"))
	show_prediction(prediction, test_images)


if __name__ == '__main__':
	main()
