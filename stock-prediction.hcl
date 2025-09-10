job "projects" {
  datacenters = ["dc1"]

  update {
    max_parallel = 1
    min_healthy_time = "10s"
    healthy_deadline = "3m"
    progress_deadline = "10m"
    auto_revert = true
    auto_promote = true
    canary = 1
  }

  constraint {
    attribute = "${attr.kernel.name}"
    value     = "linux"
  }

  group "apps" {
    count = 1
    network {
      port "stock-prediction" {
        static = 8502
        to = 8502 # Internal Port
     }
    }

    task "stock-prediction" {
      driver = "docker"
      config {
        image = "ghcr.io/csyyysc/stock-prediction"
        ports = ["stock-prediction"]
      }

      resources {
        cpu    = 128
        memory = 128
      }

      service {
        name     = "stock-prediction"
        port     = "stock-prediction"
        provider = "nomad"
        tags     = ["stock-prediction"]

        check {
          type     = "http"
          path     = "/"
          interval = "30s"
          timeout  = "5s"
          check_restart {
            limit = 3
            grace = "30s"
          }
        }
      }
    }
  }
}