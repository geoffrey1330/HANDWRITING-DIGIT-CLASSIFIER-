from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from pylab.hog import HOG
from pylab import dataset


(digits, target) = dataset.load_digits("data/digits.csv")
data = []

hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), transform = True)


for image in digits:
	
	image = dataset.deskew(image, 20)
	image = dataset.center_extent(image, (20, 20))

	# describe the image and update the data matrix
	hist = hog.describe(image)
	data.append(hist)

# train the model
model = LinearSVC(random_state = 42)
model.fit(data, target)

# dump the model to file
joblib.dump(model, "models/svm.cpickle")