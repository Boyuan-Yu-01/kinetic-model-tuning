// OpenMP schedules Python calls; each thread owns a separate Python interpreter.
#include <omp.h>
#include <spawn.h>
#include <sys/wait.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
extern char **environ;
// Executor creates a private session. Never signal a group unless this
// coordinator is its leader: standalone invocations must not kill the shell.
static void abort_group(int) {
    const pid_t self=getpid();
    if(getpgrp()==self) kill(-self,SIGKILL);
    _exit(128+SIGTERM);
}
static bool protect_parent_lifetime() {
    struct sigaction action{};
    action.sa_handler=abort_group;
    sigemptyset(&action.sa_mask);
    if(sigaction(SIGTERM,&action,nullptr)!=0 || sigaction(SIGINT,&action,nullptr)!=0)
        return false;
#ifdef __linux__
    // Install before creating workers. The expected PID also closes the race
    // where our Python parent dies before we install PR_SET_PDEATHSIG.
    const char *expected_text=std::getenv("MT_OPENMP_PARENT_PID");
    const pid_t expected=expected_text ? static_cast<pid_t>(std::strtol(expected_text,nullptr,10)) : getppid();
    if(prctl(PR_SET_PDEATHSIG,SIGTERM)!=0) return false;
    if(getppid()!=expected) abort_group(SIGTERM);
#endif
    return true;
}
struct Worker { pid_t pid=-1; FILE *input=nullptr, *output=nullptr; int tasks=0; };
static void stop(Worker &w) {
    // Called only after a result or EOF, never during an outstanding task.
    // Kill also avoids arbitrary Python atexit handlers blocking recycling.
    if(w.pid>0) kill(w.pid,SIGKILL);
    if(w.input) fclose(w.input);
    if(w.output) fclose(w.output);
    if(w.pid>0) { int status; while(waitpid(w.pid,&status,0)<0 && errno==EINTR) {} }
    w=Worker{};
}
static bool start(Worker &w, char *python, char *script) {
    bool success=false;
    // Serialize pipe creation/spawn: even on platforms lacking pipe2, another
    // spawn cannot inherit a descriptor between pipe() and CLOEXEC setup.
    #pragma omp critical(worker_spawn)
    {
        int in[2],out[2];
        if(pipe(in)==0) {
            if(pipe(out)==0) {
                for(int fd:{in[0],in[1],out[0],out[1]}) fcntl(fd,F_SETFD,FD_CLOEXEC);
                posix_spawn_file_actions_t actions; posix_spawn_file_actions_init(&actions);
                posix_spawn_file_actions_adddup2(&actions,in[0],STDIN_FILENO);
                posix_spawn_file_actions_adddup2(&actions,out[1],STDOUT_FILENO);
                char *args[]={python,script,nullptr};
                int error=posix_spawnp(&w.pid,python,&actions,nullptr,args,environ);
                posix_spawn_file_actions_destroy(&actions); close(in[0]); close(out[1]);
                if(error) { std::cerr << "Python worker spawn failed: " << std::strerror(error) << std::endl; w.pid=-1; close(in[1]); close(out[0]); }
                else {
                    w.input=fdopen(in[1],"w"); w.output=fdopen(out[0],"r");
                    success=w.input && w.output;
                    if(!w.input) close(in[1]);
                    if(!w.output) close(out[0]);
                }
            } else { close(in[0]); close(in[1]); }
        }
    }
    if(!success) stop(w);
    return success;
}
int main(int argc, char **argv) {
    if(argc!=4 && argc!=5) { std::cerr<<"usage: runner N PYTHON WORKER.py [MAX_TASKS]\n"; return 2; }
    int n=std::atoi(argv[1]); if(n<1) return 2;
    int limit=argc==5 ? std::atoi(argv[4]) : 0;
    if(!protect_parent_lifetime()) { std::cerr<<"cannot install parent lifetime protection\n"; return 3; }
    signal(SIGPIPE,SIG_IGN);
    std::vector<Worker> workers(n);
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(n)
    {
        Worker &w=workers[omp_get_thread_num()];
        for(;;) {
            std::string path; bool got=false;
            #pragma omp critical(task_input)
            { got=bool(std::getline(std::cin,path)); }
            if(!got) break;
            bool ok=(w.pid>0 || start(w,argv[2],argv[3]));
            if(ok) ok=std::fprintf(w.input,"%s\n",path.c_str())>=0 && std::fflush(w.input)==0;
            char *line=nullptr; size_t size=0;
            if(ok) ok=getline(&line,&size,w.output)>0;
            free(line);
            ++w.tasks;
            if(!ok || (limit>0 && w.tasks>=limit)) stop(w);
            #pragma omp critical(task_output)
            { std::cout<<(ok ? "OK " : "ERR ")<<path<<std::endl; }
        }
        stop(w);
    }
    return 0;
}
