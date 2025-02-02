### combine csv files

combine_csv_files <- function(folder_path) {
  # 获取文件夹下所有CSV文件的文件名
  csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
  
  # 创建一个空的列表，用于存储每个CSV文件的数据框
  df_list <- list()
  
  # 遍历每个CSV文件，读取数据并存储到df_list中，并为每个文件添加一个标识列
  for (i in 1:length(csv_files)) {
    df <- read.csv(csv_files[i])  # 读取CSV文件
    df$subjID <- i  # 添加一个名为'subjID'的列，用于表示文件编号
    df_list[[i]] <- df  # 将数据框存储到列表中
  }
  
  # 将列表中的数据框合并为一个数据框
  combined_df <- do.call(rbind, df_list)
  
  return(combined_df)
}