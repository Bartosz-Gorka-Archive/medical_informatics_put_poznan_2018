import axios from 'axios'

const api = {
  fetch (url) {
    return axios.get(url)
    .then(response => response.data)
    .catch(error => {
      console.log(error)
      return Promise.reject(error)
    })
  }
}

export default api