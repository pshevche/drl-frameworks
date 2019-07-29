#echo "going to drop cache for docker!"
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
sudo docker restart docker-pg
#echo "dropped docker pg cache"
