#!/bin/bash

# 设置目标目录。你可以修改这个变量为你实际的目录路径。
TARGET_DIR="$1"  # 从脚本的第一个参数获取目标目录
ACTION="$2"  # 从脚本的第二个参数获取保存目录


# 检查目标目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
  echo "错误: 目标目录 '$TARGET_DIR' 不存在或不是一个目录。"
  exit 1 # 退出脚本，返回错误代码 1
fi

echo "开始处理目录: $TARGET_DIR"

# 遍历目标目录下的所有子目录
find "$TARGET_DIR" -maxdepth 1 -type d -print0 | while IFS= read -r -d $'\0' subdirectory; do
  # 跳过目标目录本身 (即如果子目录就是目标目录本身)
  if [ "$subdirectory" = "$TARGET_DIR" ]; then
    continue
  fi

  echo "进入子目录: $subdirectory"

  # 切换到子目录
  cd "$subdirectory" || {
    echo "错误: 无法进入目录 '$subdirectory'。"
    continue # 继续处理下一个子目录
  }

  if [ "$ACTION" = "clean" ]; then
    if make clean; then
      echo "在 '$subdirectory' 中成功执行 make clean"
    else
      echo "在 '$subdirectory' 中执行 make clean 失败"
      # 根据需求，可选择是否退出脚本
      # exit 1
    fi
  else
    # 默认操作，执行 make dfgGen
    if make dfgGen; then
      echo "在 '$subdirectory' 中成功执行 make dfgGen"
    else
      echo "在 '$subdirectory' 中执行 make dfgGen 失败"
      # 根据需求，可选择是否退出脚本
      # exit 1
    fi
  fi

  # 返回到脚本开始执行的目录 (可选，但推荐)
  cd - > /dev/null 2>&1 || {
      echo "错误: 无法返回到原始目录，请检查 'cd -' 命令是否工作正常。"
      # 这里不退出，尽量保持脚本继续运行
  }

  echo "返回到上一级目录"
  echo "-------------------------"

done

echo "所有子目录处理完成。"
exit 0 # 脚本成功执行完成，返回代码 0