from msilib.schema import Binary
from utils import *
from tod import *
from simple_perceptron import BinaryPerceptron

#Note that WormViewer breaks if no worms are detected in an interval

# What does FALSE mean in TOD? It didn't die?

# Create list of vectors over time, look for transition to -1.

class TrainingData:
  def __init__(self, image1, image2, file_path = False):
    # Image 1 should be new
    if file_path:
      self.image1 = cv2.imread(image1,0)
      self.image1 = np.where(self.image1>128, 0., 1.)
      self.image2 = cv2.imread(image2,0)
      self.image2 = np.where(self.image2>128, 0., 1.)
    else:
      self.image1 = image1
      self.image2 = image2
      if np.max(self.image1) > 128:
        self.image1 = np.where(self.image1>128, 0., 1.)
        self.image2 = np.where(self.image2>128, 0., 1.)

  def show_images(self):
    figure, axis = plt.subplots(2)
    axis[0].imshow(self.image1)
    axis[1].imshow(self.image2)
    plt.show()

  def getData(self):
    return (self.image1, self.image2)

  def blurDifference(self,blur_v=2):
    xshape, yshape = self.image1.shape
    wormA = cv2.blur(self.image1, (blur_v, blur_v))
    wormB = cv2.blur(self.image2, (blur_v, blur_v))
    diff = cv2.absdiff(wormA, wormB)
    pixel_count = xshape * yshape

    return np.mean(diff)
  def rawDifference(self):
    if np.sum(self.image2) == 0:
      if np.sum(self.image1) == 0:
        return 0
      else:
        return 1
    return (np.sum(self.image2) - np.sum(self.image1))/np.sum(self.image2)
  def orPixels(self):
    """
    Determines what proportion of the pixels that are in either of the two images
    are in the old images

    Returns:
        float: The proportion of pixels in the old image and the OR image
    """
    shared_matrix = 1 - (1-self.image1)*(1-self.image2)
    if np.sum(shared_matrix)==0:
      return 1
    return np.sum(self.image2)/(np.sum(shared_matrix))

  def andPixels(self):
    """
    Determines what proportion of the pixels that are in the old images are shared between the two images

    Returns:
        float: The proportion of pixels in the AND image and the old image
    """
    shared_matrix = self.image1*self.image2
    if np.sum(self.image2) == 0:
      if np.sum(self.image1) == 0:
        return 1
      else:
        return 0
    return (np.sum(shared_matrix))/np.sum(self.image2)

  def getVector(self):
    return [self.rawDifference(),self.blurDifference(),self.orPixels(),self.andPixels(),self.poolDifference(4),self.noiseSensor1()]

  def biggestPool(self,image,pool_size):
    m, n = image.shape[:2]
    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    ny=_ceil(m,pool_size)
    nx=_ceil(n,pool_size)
    size=(ny*pool_size, nx*pool_size)+image.shape[2:]
    mat_pad=np.full(size,np.nan)
    mat_pad[:m,:n,...]=image
    new_shape=(ny,pool_size,nx,pool_size)+image.shape[2:]
    result= np.nansum(mat_pad.reshape(new_shape),axis=(1,3))
    return np.max(result)

  def poolDifference(self, pool_size = 5):
    return self.biggestPool(self.image2, pool_size) - self.biggestPool(self.image1,pool_size)

  def noiseSensor1(self):
    if np.sum(self.image1) > self.image1.shape[0]*self.image1.shape[1]/2:
      return 1
    else:
      return 0


def create_train_folders(csv_path, avi_path,store_path):
  video_watcher = WormViewer(csv_path,avi_path)
  worm_ids = np.arange(0, len(video_watcher.tracked))
  difs: dict[int, list] = {}
  for i in worm_ids:
      difs[i] = []

  stop = video_watcher.first - video_watcher.scan  # where to stop checking in reverse.
  start = video_watcher.first  # where to start checking in reverse.

  assert(start > stop), "Invalid scan and first params."

  specific_worm_dif = []
  skip = 20
  count = 5
  gap = 50

  for i in tqdm(range(start,stop,-skip)):
    current_worms = video_watcher.fetch_worms(worm_ids, i)
    current_worms = video_watcher.transform_all_worms(current_worms)
    #spread = count + gap
    # Sets frame range for getting worm averages.
    high = min(start, i + gap + skip * count)  # Upper bounds.
    low = min(start, i + gap)  # Lower bounds.

    # Loop through frames of interest.
    worms_to_inspect = []
    for n in range(low, high, skip):
      # wti = worm to inspect
      wti = video_watcher.fetch_worms(worm_ids, n)
      wti = video_watcher.transform_all_worms(wti)
      worms_to_inspect.append(wti)

    worms_to_inspect = np.array(worms_to_inspect, dtype=object)
    # Ignore beginning where there are no worms to compare.
    if worms_to_inspect.shape == (0,):
        print("skipping empty")
        continue
    for worm_id in worm_ids:
      older_worms = worms_to_inspect[:, worm_id]
      # older_worms = video_watcher.transform_all_worms(older_worms)
      video_watcher.older = worms_to_inspect
      current_worm = current_worms[worm_id]
      xshape, yshape = current_worm.shape

      totals = []
      for worm in older_worms:
        difference = video_watcher.calculate_difference(worm, current_worm)
        totals.append(difference.sum(axis=None))
        fold_name = store_path + "/" + str(worm_id)
        if not os.path.exists(fold_name):
          os.mkdir(fold_name)
        plt.imsave(fold_name+"/"+str(i)+str(worm_id)+"_old.png",worm)
        plt.imsave(fold_name+"/"+str(i)+str(worm_id)+"_new.png",current_worm)
        #axis[0].imshow(worm)
        #axis[1].imshow(current_worm)
        #plt.show()

      pixel_count = xshape * yshape
      avg = np.average(totals)
      avg = avg / pixel_count  # Normalize difference by pixel count.

      difs[worm_id].append(avg)

      if not video_watcher.worm_state[worm_id] and avg > video_watcher.thresh:
          video_watcher.worm_state[worm_id] = i + skip
          # included - gap to account for the fact that when the worm
          # has moved it is already alive, so go back to last time it
          # was known to be dead.fold_name+"/old",worm


if __name__=="__main__":
  #create_train_folders("C:/Users/cdkte/Downloads/videos/300/csvs/960.csv","C:/Users/cdkte/Downloads/videos/300/vids/960.avi","C:/Users/cdkte/Downloads/videos/300/train")
  """
  store_path = "C:/Users/cdkte/Downloads/videos/300/train"

  image_one = store_path+"/19/"+"0_new.png"
  image_two = store_path+"/19/"+"0_old.png"
  test = TrainingData(image_one,image_two,file_path=True)

  print(test.getVector())

  image_one = store_path+"/31/"+"0_new.png"
  image_two = store_path+"/31/"+"0_old.png"
  test = TrainingData(image_one,image_two,file_path=True)

  print(test.getVector())
  test.show_images()

  """
  perceptron = BinaryPerceptron(6,[0,0,0,0,0,0],alpha=0.01,save_path = "weights_test.csv")
  perceptron.load()

  same_images_path = "C:/Users/cdkte/Downloads/videos/300/train/train_examples/same"
  dif_images_path = "C:/Users/cdkte/Downloads/videos/300/train/train_examples/different"
  """
  train_data = []

  for file in os.listdir(same_images_path):
    file = same_images_path + "/" + file
    if len(file.split("new")) == 1:
      # do nothing on old images
      continue
    old_image = "old".join(file.split("new"))
    if not os.path.exists(old_image):
      continue
    cur_data = TrainingData(file,old_image,file_path=True)
    train_data.append((cur_data.getVector(),-1))
  for file in os.listdir(dif_images_path):
    file = dif_images_path + "/" + file
    if len(file.split("new")) == 1:
      # do nothing on old images
      continue
    old_image = "old".join(file.split("new"))
    if not os.path.exists(old_image):
      continue
    cur_data = TrainingData(file,old_image,file_path = True)
    train_data.append((cur_data.getVector(),1))

  import random as r
  r.shuffle(train_data)
  for i in range(10):
    for vector in train_data:
      perceptron.train_with_one_example(vector[0],vector[1])

  perceptron.save()
  """


  """
  count = 0
  err_count = 0
  print(perceptron.weights)
  for file in os.listdir(same_images_path):
    if len(file.split("new"))==1:
      continue
    file = same_images_path + "/" + file
    old_image = "old".join(file.split("new"))
    if not os.path.exists(old_image):
      continue
    cur_data = TrainingData(file,old_image,file_path = True)

    prediction = perceptron.classify(cur_data.getVector())

    count += 1
    if prediction!=-1:
      #print(cur_data.getVector())
      #cur_data.show_images()
      err_count += 1

  for file in os.listdir(dif_images_path):
    if len(file.split("new"))==1:
      continue
    file = dif_images_path + "/" + file
    old_image = "old".join(file.split("new"))
    if not os.path.exists(old_image):
      continue
    cur_data = TrainingData(file,old_image,file_path = True)

    prediction = perceptron.classify(cur_data.getVector())

    count+=1
    if prediction!=1:
      err_count+=1
      print(file)
      print(cur_data.getVector())
      cur_data.show_images()
  print(err_count/count,count)
  """
  img_1="C:/Users/cdkte/Downloads/videos/300/train/21/-38021_new.png"
  print(os.path.exists(img_1))
  img_2="C:/Users/cdkte/Downloads/videos/300/train/21/-38021_old.png"
  cur_data = TrainingData(img_1,img_2,file_path = True)
  print(perceptron.classify(cur_data.getVector()))
  cur_data.show_images()