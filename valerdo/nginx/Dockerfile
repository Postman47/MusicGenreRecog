# pull nginx image from Docker Hub
FROM nginx

# remove the default nginx config file and replace it with new one
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/