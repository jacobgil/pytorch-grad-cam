import cv2
import numpy as np
import matplotlib.pyplot as plt
def show_img(cam, img):

        heatmap = cv2.resize(cam.cpu().detach().numpy(), (img.shape[1], img.shape[0]))

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
 
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()