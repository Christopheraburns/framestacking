import retro
import cv2
import datetime
import os
import sys
from collections import deque
import numpy as np

now = None                      # Save the datetime
cropframe = True                # do we crop the images before training? set to True will crop
cropleft = 0                    # Number of pixels to crop from the left of the frame
croptop = 20                     # Number of pixels to crop from the top of the frame
cropright = 0                   # Number of pixels to crop from the right of the frame
cropbottom = 12                  # Number of pixels to crop from the bottom of the frame
grayscale = True                # set to true to convert frames to grayscale
env = None                      # OpenAI retro environment library
training_time = None            # value to record the training time
saveframes = False              # save the training frames to disk (saves original not cropped or grayscaled frames)
savegrayscale = False           # save the grayscale image (grayscale must be set to True)
savecropped = False             # save the cropped image (cropframe must be set to True)
MSPE = 200                      # Max steps per episode
frameheight= None
framewidth= None
stack_size = 4                  # Number of frames stacked together to counter temporal limitations



def savefile(frame, filename):
    try:
        if not os.path.exists(os.path.join(os.getcwd(), now)):
            os.mkdir(now)
        cv2.imwrite(now + '/' + filename, frame)
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in savefile function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))


def crop(frame, episode, episode_step):
    global frameheight
    global framewidth

    try:
        Y = frame.shape[0]

        X = frame.shape[1]

        y1 = croptop
        y2 = Y - cropbottom
        x1 = cropleft
        x2 = X - cropright

        cropped = frame[y1:y2, x1:x2]
        print("cropped image shape: {}".format(cropped.shape))
        cv2.imshow("Cropped", frame)
        if savecropped:
            filename = str(episode) + "_" + str(episode_step) + "_crop.jpg"
            savefile(cropped, filename)

        frameheight = Y - (croptop + cropbottom)
        framewidth = X - (cropleft + cropright)

        return cropped
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in crop function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))


def grayframe(frame, episode, episode_step):
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if savegrayscale:
            filename = str(episode) + "_" + str(episode_step) + "_gray.jpg"
            savefile(gray_frame, filename)
        return gray_frame
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in grayframe function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))


def snapframe(obs_state, episode, episode_step):
    try:
        if saveframes:
            filename = str(episode) + "_" + str(episode_step) + "_orig.jpg"
            savefile(obs_state, filename)

        cv2.imshow("Frame", obs_state)

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in snapframe function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))


def preprocessframe(obstate, episode, episode_step):
    global frameheight
    global framewidth

    try:
        # Use OpenCV to convert the numpy array to an img
        # Note - the frame exists as a cv2.numpy array for the duration of this function
        snapframe(obstate, episode, episode_step)

        frameheight = obstate.shape[0]
        framewidth = obstate.shape[1]

        # Do we convert to grayscale?
        if grayscale:
            frame = grayframe(obstate, episode, episode_step)

        # Do we crop our frames?
        if cropframe:
            frame = crop(frame, episode, episode_step)

        return frame
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in preprocessframe function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))

def stack_frames(stacked_frames, state, is_new_episode, episode, episode_step):
    global framewidth
    global frameheight

    try:
        frame = preprocessframe(state, episode, episode_step)
        if is_new_episode:
            # reset the deque object for the new episode
            stacked_frames = deque([np.zeros((frameheight, framewidth), dtype=np.int) for _ in range(stack_size)],
                                   maxlen=4)

            # a new episode means we restart with 4 copies of the initial state
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)

            stacked_state = np.stack(stacked_frames, axis=2)
        else:
            # Just add the new frame to the deque
            stacked_frames.append(frame)
            stacked_state = np.stack(stacked_frames, axis=2)


        return stacked_state, stacked_frames
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in stack_frames function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))


def start():
    global env
    global now
    global frameheight
    global framewidth

    try:
        episode_counter = 0
        episode_reward = 0
        episode_steps = 0
        # Initialize the environment engine
        env = retro.make(game='SpaceInvaders-Atari2600')

        # Debug stuff (show the # of actions and the observable space)
        '''
        action_size = env.action_space.n
        obs_space = env.observation_space
        print("observable state: {}".format(obs_space))

        print("# of Actions: {}".format(action_size))
        
        '''
        # Start a new episode
        state = env.reset()

        frameheight = state.shape[0]
        framewidth = state.shape[1]

        starttime = datetime.datetime.now()
        now = starttime.strftime("%m-%d-%Y-%H-%M")
        # Initialize "empty" stacked frames object
        stacked_frames = deque([np.zeros((frameheight, framewidth), dtype=np.int) for _ in range(stack_size)], maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, state, True, 0, 0)
        # Take random steps
        for episode in range(1):
            alive = True
            while alive:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)

                #if episode == 0:
                    #if episode_steps == 0:



                episode_reward += reward
                episode_steps += 1
                env.render()
                if done:
                    episode_counter += 1
                    print("Episode #{} steps: {}".format(episode_counter, episode_steps))
                    print("Episode #{} reward: {}".format(episode_counter, episode_reward))
                    episode_steps = 0
                    episode_reward = 0
                    env.reset()
                    now = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
                    alive = False

        stoptime = datetime.datetime.now()
        c = input("Press any key to exit")
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in start function: {} \n {} \n {} \n {}".format(err, exc_type, fname, exc_tb.tb_lineno))

start()