import time
import edgeiq
import math
"""
Use object detection to detect objects in the frame in realtime. The
types of objects detected can be changed by selecting different models.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""



def main():
    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/mobilenet_ssd")
    obj_detect.load(engine=edgeiq.Engine.DNN_OPENVINO)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                filtered_predictions = edgeiq.filter_predictions_by_label(results.predictions, ['car'])
                
                # puts only the car predictions in its own array
                car_boxes = []
                for i, prediction in enumerate(filtered_predictions):
                      car_boxes.append((i, prediction))
                # print(car_boxes)
                
                # finding the average width of the car
                avg_width = 0
                for prediction in filtered_predictions:
                  # print('label = {}, box = {}, width = {}'.format(prediction.label, prediction.box, prediction.box.width))
                   avg_width += prediction.box.width
                
                # width sorting code
                if (len(filtered_predictions) != 0):
                   avg_width = avg_width / len(filtered_predictions)
                   print(avg_width) 
                   
                    
                   for i, prediction in enumerate(filtered_predictions):
                      
                      min = i
                      
                      for j, prediction in enumerate(filtered_predictions):
                         if(filtered_predictions[j].box.start_x < filtered_predictions[min].box.start_x):
                            min = j   
                            filtered_predictions[i], filtered_predictions[min] = filtered_predictions[min], filtered_predictions[i]
                      filtered_predictions.reverse()
  
                    # for prediction in filtered_predictions:
                      # print('labelsorted = {}, boxsorted = {}, widthsorted = {}'.format(prediction.label, prediction.box, prediction.box.width))  
    
               	# distance calculation code
                spaces = 0
                for i, prediction in enumerate(filtered_predictions):                
                   if(i < (len(filtered_predictions)-1)):
                      d = abs(filtered_predictions[i+1].box.start_x - filtered_predictions[i].box.end_x)
                      c = avg_width / 5
                      n = math.floor(d/(avg_width + c))
                      spaces = spaces + n
                      # print('distance = {}, buffer = {}, number of spaces = {}, total spaces = {}'.format(d,c,n,spaces))
                      print('available parking spaces = {}'.format(spaces))

                frame = edgeiq.markup_image(
                        frame, filtered_predictions, colors=obj_detect.colors)
                
                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")
                # text.append("Distance: ".format(distance))
                

                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break
                
                
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
