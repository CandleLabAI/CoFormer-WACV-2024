import os
import numpy as np
import cv2
from tqdm import tqdm



# global_path = 'widafil_slides_data/'
# paths = [global_path+i for i in os.listdir(global_path)]


# all_p = []
# for i in paths:
#     kk = os.listdir(i)
#     for j in kk:
#         all_p.append(i +'/'+j)

# all_p.sort()

# count = 0

# for i in tqdm(all_p):
#     kk = os.listdir(i)
#     kk.sort()
#     kk = kk[:7]

#     if len(kk)==7 and kk[6][:3] == 't_t':
#         for j1, j2 in enumerate(kk[0:7]):

#             p = i+'/'+j2
            
#             if j1 == 0:
#                 img = cv2.imread(p)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('wifdal_large_crops/i_s/i_s_{}.png'.format(count), img)
                

#             elif j1 == 1:
#                 img = cv2.imread(p)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('wifdal_large_crops/i_t/i_t_{}.png'.format(count), img)


#             elif j1 == 2:
#                 img = cv2.imread(p)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('wifdal_large_crops/t_b/t_b_{}.png'.format(count), img)

#             elif j1 == 3:
#                 img = cv2.imread(p)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('wifdal_large_crops/t_f/t_f_{}.png'.format(count), img)



#             elif j1 == 4:
#                 img = cv2.imread(p)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('wifdal_large_crops/t_sk/t_sk_{}.png'.format(count), img)


#             elif j1 == 5:
#                 img = cv2.imread(p)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('wifdal_large_crops/mask_t/mask_t_{}.png'.format(count), img)
                
#             elif j1 == 6:
#                 img = cv2.imread(p)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('wifdal_large_crops/t_t/t_t_{}.png'.format(count), img)
            
#         count += 1


p = os.listdir('widafil_slides_data')
p.sort()
print(p)
for i in p:
    try:
        img = cv2.imread('widafil_slides_data/{}'.format(i)+'/i_s_{}.png'.format(i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print('widafil_slides_data/{}'.format(i)+'i_s_{}.png'.format(int(i)))
        cv2.imwrite('wifdal_large_crops/i_s/i_s_{}.png'.format(int(i)), img)

        img = cv2.imread('widafil_slides_data/{}'.format(i)+'/i_t_{}.png'.format(int(i)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('wifdal_large_crops/i_t/i_t_{}.png'.format(int(i)), img)

        img = cv2.imread('widafil_slides_data/{}'.format(i)+'/t_f_{}.png'.format(int(i)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('wifdal_large_crops/t_f/t_f_{}.png'.format(int(i)), img)

    except Exception as e:
        pass
        
