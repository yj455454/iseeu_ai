from iseeu_ai import IseeU

i = IseeU()

user_face_path = './user'
unknown_face_path = './unknown'
crop_face_path = './FD'

i.make_person_list(user_face_path, 'user')
i.make_person_list(unknown_face_path, 'unknown')
i.make_person_list(unknown_face_path, 'crop')

target, person_id = i.predict('user')

if person_id != -1:
    i.image_record_write(target, person_id)
else:
    target, person_id = i.predict('unknown')
    if person_id != -1:
        i.image_record_write(target, person_id)
    else:
        i.new_unknown_image_record_write()
