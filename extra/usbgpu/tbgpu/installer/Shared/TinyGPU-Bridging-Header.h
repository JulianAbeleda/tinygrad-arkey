// C bridge used by the command-line diagnostic path. It never starts server mode.
int run_server(const char *sock_path);
int tinygpu_keepalive_status(char *out, unsigned long out_cap, unsigned long *out_len);
int tinygpu_keepalive_handshake(char *out, unsigned long out_cap, unsigned long *out_len);
int tinygpu_power_residency_status(char *out, unsigned long out_cap, unsigned long *out_len);
