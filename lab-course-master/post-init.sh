#!/bin/bash

logmsg() {
    printf '%-19s (%06d) -- %-4s %s\n' "$(date +'%Y.%m.%d %T')" $$ "$@"
}

infomsg() {
    if (( verbose>0 )); then
        logmsg INFO "$*"
    fi
}

warnmsg() {
    logmsg WARN "$*"
}

errmsg() {
    logmsg ERR "$*"
}

### Update to latest version from upstream repository
update_from_upstream() {
    local -r org=$(git config --get 'remote.origin.url'||echo -n)
    local -r url=$(git config --get 'remote.upstream.url'||echo -n)
    local -a keys=("${UPKEY[@]:-}")

    if [[ -z "${org}" ]]; then
        errmsg "Not a git repository: $(pwd)"
        return 0
    fi
    if [[ "${org}" =~ (https?://[^/]+)/gitlab/+([^/]+)/+([^/]+)\.git$ ]]; then
        local -r odomain=${BASH_REMATCH[1]}
        local -r onamespace=${BASH_REMATCH[2]}
        local -r oproject=${BASH_REMATCH[3]}
        infomsg "origin: ${odomain}/${onamespace}/${oproject}"
    else
        warnmsg "Cannot identify project's remote origin from config ${org}"
        return 0
    fi

    if [[ -n "${UPSTREAM}" ]]; then
        #- Use provided parent URL
        local -r ups=${UPSTREAM}
    else
        #- Guess URL of parent using gitlab API, use the first key if one is provided
	if [[ -n "${keys[0]:-}" ]]; then
            local -ar header=("--header" "PRIVATE-TOKEN: ${keys[0]}")
	    keys=(${keys[@]:1})
        else
            local -ar header
        fi 
        local m=$(curl -s "${header[@]}" "${odomain}/api/v4/projects?search_namespaces=true&search=${onamespace}%2F${oproject}")
        local r=$(echo "${m}"|jq '.[]|.forked_from_project|.path_with_namespace' || echo -n)
        if [[ "${r}" =~ ^[^/]+/[^/]+$ ]]; then
            local -r ups=${odomain}/${r//\"/}.git
        else
            warnmsg "Cannot retrieve project's remote parent from gitlab"
            warnmsg "${m}"
            return 0
        fi
    fi

    if [[ "${ups}" =~ (https?://[^/]+)?/?([^/]+)/+([^/]+)\.git$ ]]; then
        local -r udomain=${BASH_REMATCH[1]:-${odomain}}
        local -r unamespace=${BASH_REMATCH[2]}
        local -r uproject=${BASH_REMATCH[3]}
    else
        warnmsg "Cannot identify project's remote parent from ${ups}"
        return 0
    fi

    if [[ "${unamespace}" == "${onamespace}" && "${uproject}" == "${oproject}" ]]; then
        warnmsg "Will not set upstream for parent repo: ${org}"
        return 0
    fi

    #- Compose parent's URL, use the remaining key if one is provided
    if [[ "${udomain}" =~ (https?://)?(oauth:[^@]+@)?(.+)$ ]]; then
	local -r uproto=${BASH_REMATCH[1]}
	local -r udname=${BASH_REMATCH[3]}
	if [[ -n "${BASH_REMATCH[2]}" ]]; then
	    local -r uoauth=${BASH_REMATCH[2]}
	else
	    if [[ "${keys[0]:-x}" != "x" ]]; then
	        local -r uoauth="oauth:${keys[0]}@"
	        keys=(${keys[@]:1})
	    else
	        local -r uoauth=''
	    fi
	fi
    else
        warnmsg "Invalid URL for project's remote parent ${ups}"
	return 0
    fi
    local -r upstream=${uproto}${uoauth}${udname}/gitlab/${unamespace}/${uproject}.git
    if [[ "${onamespace}/${oproject}" != "com490-teachers/com490-2022" ]]; then

      if [[ -n "${url}" ]]; then
          warnmsg "Replacing current remote.upstream.url: ${url}."
          git remote remove upstream
      fi

      #- Update both code and LFS files from parent stream, keep trying for 120s if needed
      infomsg "Update repository from upstream: ${upstream}"
      git remote add upstream "${upstream}"
      local -i starttime=$(date +%s)
      while (( $(date +%s) < starttime+120 )) && ! errmsg=$(git pull --no-edit upstream master 2>&1); do
          warnmsg "Git pull was unsuccessful"
	  warnmsg "Error: ${errmsg}"
	  warnmsg "Will try again in 5s"
          sleep 5
      done
      while (( $(date +%s) < starttime+120 )) && ! errmsg=$(git lfs pull 2>&1); do
          warnmsg "Git LFS pull was unsuccessful"
	  warnmsg "Error: ${errmsg}"
	  warnmsg "Will try again in 5s"
          sleep 5
      done
      if ! git push origin master; then
          warnmsg "Git push failed"
      fi
      if ! git lfs pull upstream master; then
          warnmsg "Git LFS pull failed"
      fi
    fi
    return 0
}

cleanup() {
    infomsg "Goodbye"
}

setup_jupytext() {
    # Setup jupyter config to open md and py files as notebooks by default.
    mkdir -p ${HOME}/.jupyter/labconfig
    rm -f ${HOME}/.jupyter/labconfig/default_setting_overrides.json
    cat << STR > ${HOME}/.jupyter/labconfig/default_setting_overrides.json
{
  "@jupyterlab/docmanager-extension:plugin": {
    "defaultViewers": {
      "markdown": "Jupytext Notebook",
      "python": "Jupytext Notebook"
    }
  }
}
STR
}

main() {
    set -ueC

    # You can hardcode the parent project into UPSTREAM, or leave this variable empty.
    # Use the UPKEY array variable to hold a list of access keys.
    # The keys are consumed from UPKEY in the following order:
    #   - gitlab *read-api* key, used to query the parent project URL, only if UPSTREAM is empty, 
    #   - git *read-repository* key, only if the parent project is private
    #
    local -r  UPSTREAM="com490/lab-course.git"
    local -ar UPKEY=()

    local -r  src=$(readlink -en ${BASH_SOURCE[0]})
    local -r  dir=$(dirname "${src}")
    local -r  cwd=$(pwd)
    local -r  cmd=$(basename "${src}")
    local -r  md5=$(md5sum "${src}")
    local -i  rerun=0
    local -i  front=0
    local -ig verbose=1

    while getopts "fu" ARGV; do
        case "${ARGV}" in
            'f') front+=1 ;;
            'u') rerun+=1 ;;
        esac
    done

    #- Get a file lock, garantees only one instance of this command runs
    exec {fdlock}<"${src}"
    if ! flock -x -w 5 -E 1 ${fdlock}; then
        errmsg "Another ${cmd} is running, try again later"
        return 1
    fi

    if (( front == 0 )); then
        if (( $(pgrep -fc bash) < 10 )); then
            infomsg "Daemonize, see logs in /tmp/post-init.log"
            ( ${src} -f "${@}" & ) &
        else
            errmsg "Too many threads"
        fi
        flock -u ${fdlock}
        return 0
    fi

    exec >>/tmp/post-init.log 2>&1
    trap cleanup EXIT

    infomsg "${rerun}: ${md5}"

    cd "${dir}"
    update_from_upstream
    cd "${cwd}"

    if ! md5sum --status -c <(echo "${md5}"); then
        if (( rerun < 3 )); then
            warnmsg "Run updated post-init.sh with -u (update) argument."
            exec ${src} "$@" -u
        fi
    fi
    setup_jupytext

    return 0
}

main "${@}"

