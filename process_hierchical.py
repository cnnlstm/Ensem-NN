#coding=utf-8
import sys
import os
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import cv2
import math
import h5py




classes = {
    0: 'drink water',
    1: 'eat meal/snack',
    2: 'brushing teeth',
    3: 'brushing hair',
    4: 'drop',
    5: 'pickup',
    6: 'throw',
    7: 'sitting down',
    8: 'standing up (from sitting position)',
    9: 'clapping',
    10: 'reading',
    11: 'writing',
    12: 'tear up paper',
    13: 'wear jacket',
    14: 'take off jacket',
    15: 'wear a shoe',
    16: 'take off a shoe',
    17: 'wear on glasses',
    18: 'take off glasses',
    19: 'put on a hat/cap',
    20: 'take off a hat/cap',
    21: 'cheer up',
    22: 'hand waving',
    23: 'kicking something',
    24: 'put something inside pocket / take out something from pocket',
    25: 'hopping (one foot jumping)',
    26: 'jump up',
    27: 'make a phone call/answer phone',
    28: 'playing with phone/tablet',
    29: 'typing on a keyboard',
    30: 'pointing to something with finger',
    31: 'taking a selfie',
    32: 'check time (from watch)',
    33: 'rub two hands together',
    34: 'nod head/bow',
    35: 'shake head',
    36: 'wipe face',
    37: 'salute',
    38: 'put the palms together',
    39: 'cross hands in front (say stop)',
    40: 'sneeze/cough',
    41: 'staggering',
    42: 'falling',
    43: 'touch head (headache)',
    44: 'touch chest (stomachache/heart pain)',
    45: 'touch back (backache)',
    46: 'touch neck (neckache)',
    47: 'nausea or vomiting condition',
    48: 'use a fan (with hand or paper)/feeling warm',
    49: 'punching/slapping other person',
    50: 'kicking other person',
    51: 'pushing other person',
    52: 'pat on back of other person',
    53: 'point finger at the other person',
    54: 'hugging other person',
    55: 'giving something to other person',
    56: 'touch other persons pocket',
    57: 'handshaking',
    58: 'walking towards each other',
    59: 'walking apart from each other'
    }

training_subjects = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
_EPS = np.finfo(float).eps * 4.0
feat_dim = 150 
#traversal_order = [9,10,11,12,25,24,24,25,12,11,10,9,5,6,7,8,23,22,22,23,8,7,6,5,4,3,21,2,1,1,2,21,3,4,17,18,19,20,20,19,18,17,13,14,15,16,16,15,14,13]
traversal_order = [9,10,11,12,25,24,24,25,12,11,10,9,5,6,7,8,23,22,22,23,8,7,6,5,4,3,21,2,1,17,18,19,20,20,19,18,17,13,14,15,16,16,15,14,13,1,2,21,3,4]


leftup = [9,10,11,12,25,24]
rightup = [5,6,7,8,23,22]
leftdown = [17,18,19,20]
rightdown = [13,14,15,16]
middle = [1,2,21,3,4]


def vids_with_missing_skeletons():
  f = open("/home/siat/xyy/dataset/nturgbd/samples_with_missing_skeletons.txt",'r')
  bad_files = []
  for line in f:
    bad_files.append(line.strip()+'.skeleton')
  f.close()
  return bad_files

def generate_data(argv):
  w = 0
  bad_files = vids_with_missing_skeletons()
  skeleton_dir_root = "/data2/backup/nturgbd_data/nturgbd_skeleton/ordinary_skeletons/"
  skeleton_files = os.listdir(skeleton_dir_root)
  data_out_dir = "/data2/backup/nturgbd_data/nturgbd_skeleton/hier/hier_2/"

  #sk_info = {} # key: file_name, value: corresponding vid_info dict
  aasa = []
  max_vid_length = -1
  X_train = []
  X_test = []
  y_train = []
  y_test = []
  n_classes = 60

  num_files = len(skeleton_files)
  count = 0
  for file_name in skeleton_files:
    if file_name in bad_files:
      continue
    
    action_class = int(file_name[file_name.find('A')+1:file_name.find('A')+4])-1
    subject_id = int(file_name[file_name.find('P')+1:file_name.find('P')+4])
    # if action_class == 0:
    #   print "action_class:",action_class
    one_hot = np.zeros(n_classes)
    one_hot[action_class] = 1
    #print one_hot
    #print "subject_id:",subject_id

    sf = open(os.path.join(skeleton_dir_root,file_name),'r')
    num_frames = int(sf.readline())
    #feature = np.zeros((num_frames, feat_dim))   #[帧数,150]
    
    video_arr = np.zeros((300,5,36))
    for n in range(0,num_frames):
      body_count = int(sf.readline())
      if body_count > 2:
        for b in range(0,body_count):
          body_info = sf.readline()
          joint_count = int(sf.readline())
          for j in range(0,joint_count):
            joint_info = sf.readline()
      else: 
        #print "body_count:",body_count   
        binfo = dict()
        norm_dist = 0
        anchor = None
        right_to_left = None
        spine_to_top = None
        

        body_arr = np.zeros((5,36))
        for b in range(0,body_count):
          
          body_info = sf.readline()
          #print "body_info:",body_info
          bsp = body_info.split()
          #print "bsp:",bsp
          body_id = bsp[0]
          #print "body_id:",body_id
          cliped_edges = bsp[1]
          #print "cliped_edges:",cliped_edges
          lefthand_confidence = bsp[2]
          #print "lefthand_confidence:",lefthand_confidence
          lefthand_state = bsp[3]
          #print "lefthand_state:",lefthand_state
          righthand_confidence = bsp[4]
          #print "righthand_confidence:",righthand_confidence
          righthand_state = bsp[5]
          #print "righthand_state:",righthand_state
          is_restricted = bsp[6]
          #print "is_restricted:",is_restricted
          lean_x = bsp[7]
          #print "lean_x:",lean_x
          lean_y = bsp[8]
          #print "lean_y:",lean_y
          body_tracking_state = bsp[9]
          #print "body_tracking_state:",body_tracking_state

          #binfo[b] = bsp
          joint_count = int(sf.readline()) ## ASSUMING THIS IS ALWAYS 25
          #print "joint_count:",joint_count

          jinfo = []#dict()
          
          for j in range(0,joint_count):
            joint_info = sf.readline()
            jsp = joint_info.split()
            x = float(jsp[0])
            y = float(jsp[1])
            z = float(jsp[2])
            coord = np.zeros(3)
            coord[0]=x
            coord[1]=y 
            coord[2]=z 
            #print coord



            jinfo.append(coord) #jinfo = [25:7]
          # END JOINT LOOP
          left_arm = []
          for order in range(len(leftup)):
            left_arm.append(jinfo[leftup[order]-1])
          left_arm = np.array(left_arm).flatten()

          right_arm = []
          for order in range(len(rightup)):
            right_arm.append(jinfo[rightup[order]-1])
          right_arm = np.array(right_arm).flatten()
          
          left_leg = []
          for order in range(len(leftdown)):
            left_leg.append(jinfo[leftdown[order]-1])
          left_leg = np.array(left_leg).flatten()
          
          right_leg = []
          for order in range(len(rightdown)):
            right_leg.append(jinfo[rightdown[order]-1])
          right_leg = np.array(right_leg).flatten()
          
          torso = []
          for order in range(len(middle)):
            torso.append(jinfo[middle[order]-1])
          torso = np.array(torso).flatten()
          if b == 0:
            body_arr[0,:18] = left_arm
            body_arr[1,:18] = right_arm
            body_arr[2,:12] = left_leg
            body_arr[3,:12] = right_leg
            body_arr[4,:15] = torso
          else:
            body_arr[0,18:36] = left_arm
            body_arr[1,18:36] = right_arm
            body_arr[2,18:30] = left_leg
            body_arr[3,18:30] = right_leg
            body_arr[4,18:33] = torso



          #print body_arr.shape
          video_arr[n,:,:] = body_arr
          #body_arr.append()
          #print video_arr#.shape
          






          # traversal_arr = []
          # for order in range(len(traversal_order)):
          #   #print jinfo[traversal_order[order]-1]
          #   traversal_arr.append(jinfo[traversal_order[order]-1])

          
          # traversal_arr = np.array(traversal_arr)
          # if os.path.isdir(data_out_dir+file_name.split(".")[0]+'/'+str(n)+'_frame/') == False:
          #   os.makedirs(data_out_dir+file_name.split(".")[0]+'/'+str(n)+'_frame/')
          # arr_path = data_out_dir+file_name.split(".")[0]+'/'+str(n)+'_frame/'+'body_'+str(b)+'.h5'
          # print arr_path
          # f = h5py.File(arr_path,'w')   
          # f['data'] = traversal_arr
          # f["label"] = one_hot
          # f.close() 
          # print traversal_arr.shape
          # arr_num = arr_num+1
          # print arr_num

    sf.close()
    count +=1
    print  count
    video_arr = np.transpose(video_arr,(1,0,2))
    print video_arr.shape
    arr_path = data_out_dir+file_name.split(".")[0]+'.h5'
    f = h5py.File(arr_path,'w')   
    f['data'] = video_arr
    f["label"] = one_hot
    f.close() 


def generate_train():
  i = 0
  data_out_dir = '/home/siat/xyy/dataset/nturgbd/subjects_split_demo/'
  training_subjects = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
  
  
  for file_dir in os.listdir(data_out_dir):
    subject_id = int(file_dir[file_dir.find('P')+1:file_dir.find('P')+4])
    # for frame in os.listdir(data_out_dir+"/"+file_dir):
    #     for body in os.listdir(data_out_dir+'/'+file_dir+"/"+frame):
    #       i = i+1
    #       print i
    if subject_id in training_subjects:
      f = open("demo_train_video_list.txt","a")
      f.write(str(file_dir))#+'/'+frame+'/'+body))
      f.write("\n")
      f.close()
    else:
      g = open("demo_test_video_list.txt","a")
      g.write(str(file_dir))#+'/'+frame+'/'+body))
      g.write("\n")
      g.close()


generate_data(sys.argv)





#if __name__ == "__main__":
#  generate_data(sys.argv)
