from multiprocessing import Pool
from dotenv import load_dotenv
import os, re, time, json
import stat, glob


def get_rootpath():
    """
    获取工作区的根目录(本文件的上一层目录)
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_env(env_dict: dict[str, str] = {}):
    """
    从本地的env文件加载.env文件

    Parameters:
        env_dict: 要读取的环境变量字典与对应的默认值
    Retuens:
        读取的键值对字典
    """
    root_path = get_rootpath()
    os.chdir(root_path)
    if not load_dotenv(f"{root_path}/.env", override=True):
        print(f"load .env error, skip it: {root_path}/.env")
    if env_dict is None or not isinstance(env_dict, dict):
        env_dict = {}
    for key in env_dict:
        read_value = str(os.getenv(key))
        env_dict[key] = read_value if len(read_value) > 0 else env_dict[key]
    return env_dict


def get_size(size: "float|int"):
    """
    将Byte单位的Size转为带有单位的字符串

    Parameters:
        size: 数据大小, 单位为Byte
    Retuens:
        带有单位的数据大小, 例如: 10MB
    """
    units = ["B", "kB", "MB", "GB", "TB"]
    unit_index = 0
    while size > 1024:
        size = size / 1024.00
        unit_index += 1
    return f"{size}{units[unit_index]}"


def get_script(template_path: str, args_dict: dict, script_path: "object|str" = None):
    """
    从模板文件中获取脚本文件, 并将模板文件中的相关变量替换为指定变量

    Parameters:
        template_path: 模板文件的路径
        value_dict: 批量替换的参数字符串列表
        script_path: 修改后的脚本文件储存路径
    Returns:
        返回修改后的脚本文件内容
    """
    with open(template_path, "r+", encoding="utf-8") as file:
        lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i]
        index = line.find("#")
        lines[i] = line[:index] if index >= 0 else line
    context = "".join(lines)
    for key in args_dict:
        value = str(args_dict[key])
        key = "{" + str(key) + "}"
        while key in context:
            context = context.replace(key, value)
    if not script_path is None and isinstance(script_path, str):
        save_path = os.path.dirname(script_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(script_path, "w+", encoding="utf-8") as file:
            file.writelines(context)
        os.chmod(script_path, stat.S_IRWXU)
    return context.replace("\\", " ").replace("\r", " ").replace("\n", " ")


def get_time(start_time: float):
    """
    获取当前时间到start_time的时间差, 并返回字符串格式的时间

    Parameters:
        start_time: 开始的时间, time.time()
    Returns:
        返回当前的时间差
    """
    second = int(time.time() - start_time)
    (minute, second) = (second // 60, second % 60)
    (hour, minute) = (minute // 60, minute % 60)
    (day, hour) = (hour // 24, hour % 24)
    return (f"{day} day " if day > 0 else "") + f"{hour:02d}:{minute:02d}:{second:02d}"


def __modify_timeline_json(timeline_path, device):
    """
    修改timeline json文件, 添加设备信息

    Parameters:
        timeline_path: timeline文件所在的路径
        device: 添加的设备名称
    """
    with open(timeline_path, "r+", encoding="utf-8") as file:
        json_data = json.load(file)
    for node in json_data:
        if node["name"] in ["process_name", "thread_name"]:
            old_name = node["args"]["name"]
            node["args"]["name"] = f"{device} {old_name}"
    return json_data


def __save_timeline_json(timeline_dict):
    """
    保存timeline的json文件, 为了方便多进程处理需要单独取出内容

    Paramters:
        timeline_dict: 传入的处理参数
            timeline_dict = {
                "key": "path/to/save/timeline.json",
                "value": []
            }
    """
    timeline_path = timeline_dict["key"]
    save_path = os.path.dirname(timeline_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(timeline_path, "w+", encoding="utf-8") as file:
        json.dump(timeline_dict["value"], file)


def merge_timeline(src_path: list, dst_path: str):
    """
    合并生成的Timeline Json的文件

    Parameters:
        src_path: 每个Rank的Timeline Json所在的文件路径
        dst_path: 合并后的Json文件保存路径
    Returns:
        返回合并后的文件路径
    """
    json_data = []
    for timeline_path in src_path:
        device_path = glob.glob(f"{timeline_path}/device_*")[0]
        device = re.search(r"device_\d+", device_path)
        device_id = -1 if device is None else int(device.group().split("_")[-1])
        timeline_path = glob.glob(f"{timeline_path}/timeline/msprof_*.json")[-1]
        json_data.extend(__modify_timeline_json(timeline_path, f"device_{device_id}"))
    tasks = {}
    pid_tasks = {}
    pool_tasks = []
    for node in json_data:
        if node["name"] in ["process_name"]:
            task = " ".join(node["args"]["name"].split(" ")[1:])
            pid = str(node["pid"])
            pid_tasks[pid] = task
            if not task in tasks:
                tasks[task] = []
    for node in json_data:
        pid = str(node["pid"])
        task = pid_tasks[pid]
        tasks[task].append(node)
    for task in tasks:
        timeline_name = task.replace(" ", "_").lower()
        pool_tasks.append(
            {
                "key": f"{dst_path}/{timeline_name}.json",
                "value": tasks[task],
            }
        )
    with Pool(len(pool_tasks)) as process:
        process.map(__save_timeline_json, pool_tasks)
