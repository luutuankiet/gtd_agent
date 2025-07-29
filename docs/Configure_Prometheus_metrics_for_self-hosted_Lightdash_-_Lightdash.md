# Configure Prometheus metrics for self-hosted Lightdash - Lightdash

**Source:** https://docs.lightdash.com/self-host/customize-deployment/configure-prometheus-metrics-for-self-hosted-lightdash

Lightdash home page
Search...
Ctrl KAsk AI
Search...
Navigation
Customize deployment
Configure Prometheus metrics for self-hosted Lightdash
##### Introduction
  * Welcome to Lightdash
  * Setting up a new project


##### Explore and analyze
  * Quickstart
  * Guides
  * Reference


##### Build your semantic layer
  * Developer quickstart
  * CLI quickstart
  * Guides
  * Reference


##### Workspace and user management
  * Guides
  * Reference


##### Integrations


##### Embedding and SDKs
  * Embedding quickstart
  * Embedding reference


##### Self-hosting and deployment
  * Self-Host Lightdash
  * Lightdash Cloud vs. Self-Hosted
  * Updating Lightdash
  * Customize deployment
    * Standalone scheduler worker
    * Environment variables
    * Resource recommendations
    * Secure Lightdash with HTTPS


##### Contact


On this page
  * Enabling Prometheus metrics
  * Configuration options
  * Available metrics
  * Process metrics
  * Node.js metrics
  * PostgreSQL metrics
  * Queue metrics
  * Using metrics for monitoring and alerting
  * Setting up a Prometheus server
  * General Prometheus setup
  * Setting up Prometheus in Google Cloud Platform (GCP)
  * Setting up Prometheus in Amazon Web Services (AWS)


Lightdash can expose Prometheus metrics to help you monitor the performance and health of your Lightdash instance. This guide explains how to enable and configure Prometheus metrics for your self-hosted Lightdash deployment.
##  Enabling Prometheus metrics
By default, Prometheus metrics are disabled in Lightdash. To enable them, set the following environment variable:
Copy
Ask AI
```
LIGHTDASH_PROMETHEUS_ENABLED=true

```

##  Configuration options
You can customize the Prometheus metrics endpoint using the following environment variables: Variable | Description | Required? | Default  
---|---|---|---  
`LIGHTDASH_PROMETHEUS_ENABLED` | Enables/Disables Prometheus metrics endpoint | `false`  
`LIGHTDASH_PROMETHEUS_PORT` | Port for Prometheus metrics endpoint | `9090`  
`LIGHTDASH_PROMETHEUS_PATH` | Path for Prometheus metrics endpoint | `/metrics`  
`LIGHTDASH_PROMETHEUS_PREFIX` | Prefix for metric names  
`LIGHTDASH_GC_DURATION_BUCKETS` | Buckets for duration histogram in seconds | `0.001, 0.01, 0.1, 1, 2, 5`  
`LIGHTDASH_EVENT_LOOP_MONITORING_PRECISION` | Precision for event loop monitoring in milliseconds. Must be greater than zero.  
`LIGHTDASH_PROMETHEUS_LABELS` | Labels to add to all metrics. Must be valid JSON  
##  Available metrics
Lightdash exposes the following metrics:
###  Process metrics
These metrics provide information about the Node.js process running Lightdash: Metric | Type | Description  
---|---|---  
`process_cpu_user_seconds_total` | counter | Total user CPU time spent in seconds  
`process_cpu_system_seconds_total` | counter | Total system CPU time spent in seconds  
`process_cpu_seconds_total` | counter | Total user and system CPU time spent in seconds  
`process_start_time_seconds` | gauge | Start time of the process since unix epoch in seconds  
`process_resident_memory_bytes` | gauge | Resident memory size in bytes  
`process_virtual_memory_bytes` | gauge | Virtual memory size in bytes  
`process_heap_bytes` | gauge | Process heap size in bytes  
`process_open_fds` | gauge | Number of open file descriptors  
`process_max_fds` | gauge | Maximum number of open file descriptors  
###  Node.js metrics
These metrics provide information about the Node.js runtime: Metric | Type | Description  
---|---|---  
`nodejs_eventloop_lag_seconds` | gauge | Lag of event loop in seconds  
`nodejs_eventloop_lag_min_seconds` | gauge | The minimum recorded event loop delay  
`nodejs_eventloop_lag_max_seconds` | gauge | The maximum recorded event loop delay  
`nodejs_eventloop_lag_mean_seconds` | gauge | The mean of the recorded event loop delays  
`nodejs_eventloop_lag_stddev_seconds` | gauge | The standard deviation of the recorded event loop delays  
`nodejs_eventloop_lag_p50_seconds` | gauge | The 50th percentile of the recorded event loop delays  
`nodejs_eventloop_lag_p90_seconds` | gauge | The 90th percentile of the recorded event loop delays  
`nodejs_eventloop_lag_p99_seconds` | gauge | The 99th percentile of the recorded event loop delays  
`nodejs_active_resources` | gauge | Number of active resources that are currently keeping the event loop alive, grouped by async resource type  
`nodejs_active_resources_total` | gauge | Total number of active resources  
`nodejs_active_handles` | gauge | Number of active libuv handles grouped by handle type  
`nodejs_active_handles_total` | gauge | Total number of active handles  
`nodejs_active_requests` | gauge | Number of active libuv requests grouped by request type  
`nodejs_active_requests_total` | gauge | Total number of active requests  
`nodejs_heap_size_total_bytes` | gauge | Process heap size from Node.js in bytes  
`nodejs_heap_size_used_bytes` | gauge | Process heap size used from Node.js in bytes  
`nodejs_external_memory_bytes` | gauge | Node.js external memory size in bytes  
`nodejs_heap_space_size_total_bytes` | gauge | Process heap space size total from Node.js in bytes  
`nodejs_heap_space_size_used_bytes` | gauge | Process heap space size used from Node.js in bytes  
`nodejs_heap_space_size_available_bytes` | gauge | Process heap space size available from Node.js in bytes  
`nodejs_version_info` | gauge | Node.js version info  
`nodejs_gc_duration_seconds` | histogram | Garbage collection duration by kind  
`nodejs_eventloop_utilization` | gauge | The calculated Event Loop Utilization (ELU) as a percentage  
###  PostgreSQL metrics
These metrics provide information about the PostgreSQL connection pool: Metric | Type | Description  
---|---|---  
`pg_pool_max_size` | gauge | Max size of the PG pool  
`pg_pool_size` | gauge | Current size of the PG pool  
`pg_active_connections` | gauge | Number of active connections in the PG pool  
`pg_idle_connections` | gauge | Number of idle connections in the PG pool  
`pg_queued_queries` | gauge | Number of queries waiting in the PG pool queue  
`pg_connection_acquire_time` | histogram | Time to acquire a connection from the PG pool in milliseconds  
`pg_query_duration` | histogram | Histogram of PG query execution time in milliseconds  
###  Queue metrics
Metric | Type | Description  
---|---|---  
`queue_size` | gauge | Number of jobs in the queue  
##  Using metrics for monitoring and alerting
You can use these metrics to create dashboards and alerts in your monitoring system. Some common use cases include:
  * Monitoring memory usage and setting alerts for potential memory leaks
  * Tracking PostgreSQL connection pool utilization
  * Monitoring event loop lag to detect performance issues
  * Setting up alerts for high CPU usage

For example, you might want to create alerts for:
  * High memory usage: `process_resident_memory_bytes > threshold`
  * Event loop lag: `nodejs_eventloop_lag_p99_seconds > threshold`
  * Database connection pool saturation: `pg_active_connections / pg_pool_max_size > 0.8`


##  Setting up a Prometheus server
If you don’t already have a Prometheus server set up, here are some resources to help you get started:
###  General Prometheus setup
  * Prometheus Getting Started Guide - Official documentation on how to install and configure Prometheus
  * Prometheus Installation - Different ways to install Prometheus
  * Prometheus Configuration - Detailed configuration options for Prometheus


###  Setting up Prometheus in Google Cloud Platform (GCP)
  * Google Cloud Managed Service for Prometheus - Google Cloud’s managed Prometheus service
  * Installing Prometheus on GKE - Setting up Prometheus on Google Kubernetes Engine
  * Google Cloud Operations Suite Integration - Integrating Prometheus with Google Cloud Operations Suite


###  Setting up Prometheus in Amazon Web Services (AWS)
  * Amazon Managed Service for Prometheus - AWS managed Prometheus service
  * Getting Started with Amazon Managed Service for Prometheus - Official AWS documentation
  * Setting up Prometheus on Amazon EKS - Deploying Prometheus on Amazon Elastic Kubernetes Service


Assistant
Responses are generated using AI and may contain mistakes.


