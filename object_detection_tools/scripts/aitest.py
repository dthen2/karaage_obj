import id_network as ids
import object_detection_ids3 as obj
import time

if __name__ == '__main__':
    # prepare ID_network
    idnet = ids.ID_NetWork()
    project_ID = idnet.register_ID({"data_type": "project"}) # 1
    app_ID =     idnet.register_ID({"data_type": "app"})     # 2
    work_ID =    idnet.register_ID({"data_type": "work"})    # 3
    projects = {"data_type": "point", "project_ID": project_ID, "app_ID": app_ID, "work_ID": work_ID}
    # prepare ID_network end
    writer = obj.object_detection_init()
    object_list_previous = []
    while True:
        status, object_list_previous = obj.object_detection(writer, idnet, projects, object_list_previous, last_time=time.time())
        if status == False:
            break
    obj.object_detection_end(writer)
