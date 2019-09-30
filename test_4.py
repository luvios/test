                        t7=time.time()
                        single_person_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track.track_id]

                        match_ok,current_wear_list = person_status_match(single_person_box, other_boxs, current_worker_status)

                        add_flag = False
                        note_violation_behavior(track.track_id,current_wear_list)
                        if track.track_id in ABNORMAL_DETECT_STATUS_DELTA.keys() and ABNORMAL_DETECT_STATUS_DELTA[track.track_id]['count']==EVNENT_NOTE_COUNT_THRESH:
                            add_flag, current_wear_list, event_id_list = status_change_note(befor_worker_status, current_worker_status, track.track_id)
                            update_flag = True
                        print(ABNORMAL_DETECT_STATUS_DELTA)
                        #add_flag, current_wear_list, event_id_list = status_change_note(befor_worker_status, current_worker_status, track.track_id)
                        #t8=time.time()
                        #print('match cost:',t8-t7)
                        #print(current_wear_list)
                        #print(befor_worker_status,current_worker_status)

                        if float(int(frame.shape[0]) - bbox[3]) / float(frame.shape[0]) >= CLIMB_FLAG:
                            # 未带安全帽
                            t10 = time.time()
                            if add_flag or 0 in current_wear_list:
                                print('warning:climb')
                                frame1 = frame1.copy()
                                #cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_red,2)
                                unwear=[gl.DETECT_TYPE[i] for i in range(len(current_wear_list)) if current_wear_list[i]==NOT_WEAR_FLAG]
                                label=str(track.track_id)+' WARNING:['+",".join(unwear)+']'

                                #cv2.putText(frame1, str(track.track_id),
                                cv2.putText(frame1, label,
                                            (int((bbox[0] + bbox[2]) / 2), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            color_red,
                                            2)
                                """
                                wear_list = current_worker_status[track.track_id]

                                for i in range(len(wear_list)):
                                    wear_box = other_boxs[wear_list[i][1]]
                                    cv2.rectangle(frame1, (int(wear_box[0]), int(wear_box[1])),
                                                  (int(wear_box[2]), int(wear_box[3])), color_white, 2)
                                """
                                if add_flag and update_flag is True:
                                                                                                                                        561,1         81%
