import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  patients: [],
  loadingPatients: true,
  loadingObservations: true,
  selectedPatient: null,
  observations: [],
  totalVersions: 1,
  familyName: '',
  date: '',
  patientID: '',
  dataGraph: [
    [],
    []
  ],
  patientDetailsURL: constPaths.PATIENT_URL,
  nextUrl: constPaths.PATIENT_URL
}

const getters = {
  patients: state => state.patients,
  loadingPatients: state => state.loadingPatients,
  loadingObservations: state => state.loadingObservations,
  familyName: state => state.familyName,
  selectedPatient: state => state.selectedPatient,
  totalVersions: state => state.totalVersions,
  observations: state => state.observations,
  dataGraph: state => state.dataGraph
}

const actions = {
  getPatients ({ state, commit }) {
    return api.fetch(state.nextUrl)
    .then(data => {
      commit('setList', data)
      return state.patients
    })
    .catch(error => Promise.reject(error))
  },
  getSinglePatient ({ state, commit }, patientID) {
    return api.fetch(constPaths.PATIENT_URL + patientID)
    .then(data => {
      commit('setSelectedPatient', data)
      return state.selectedPatient
    })
    .catch(error => Promise.reject(error))
  },
  getPatientObservations ({ state, commit }) {
    return api.fetch(state.patientDetailsURL)
    .then(data => {
      commit('setObservation', data)
      return state.observations
    })
    .catch(error => Promise.reject(error))
  },
  updatePatient ({ state, commit }, { birthDate, patientID, gender }) {
    var content = state.selectedPatient
    content.birthDate = birthDate
    content.gender = gender
    return api.update(constPaths.PATIENT_URL + patientID, content)
    .then(data => {
      return api.fetch(constPaths.PATIENT_URL + patientID)
      .then(data => {
        commit('setSelectedPatient', data)
        return state.selectedPatient
      })
      .catch(error => Promise.reject(error))
    })
    .catch(error => Promise.reject(error))
  },
  getSingleVersionedPatient ({ state, commit }, { patientID, versionNumber }) {
    return api.fetch(constPaths.PATIENT_URL + patientID + '/_history/' + versionNumber)
    .then(data => {
      commit('setSelectedPatient', data)
      return state.selectedPatient
    })
    .catch(error => Promise.reject(error))
  },
  setFindByFamilyName ({ state, commit }, name) {
    commit('setFamilyNameFilter', name)
  },
  setDateFilter ({ state, commit }, date) {
    commit('filterByDate', date)
    return state.observations
  },
  clearDateFilter ({ state, commit }) {
    commit('clearDateFilter')
    return state.observations
  },
  clear ({ state, commit }) {
    commit('clear')
    return state.patients
  }
}

const mutations = {
  setFamilyNameFilter (state, name) {
    state.patients = []
    state.familyName = name
    state.loadingPatients = true
    state.nextUrl = constPaths.PATIENT_URL + '?family=' + name
  },
  setSelectedPatient (state, data) {
    state.selectedPatient = data
    state.observations = []
    state.dataGraph[0] = []
    state.dataGraph[1] = []
    state.patientDetailsURL = constPaths.PATIENT_URL + data.id + '/$everything?_sort_by=date'
    state.loadingObservations = true
    state.patientID = data.id
    state.totalVersions = parseInt(data.meta.versionId)
  },
  setObservation (state, data) {
    state.dataGraph[0] = []
    state.dataGraph[1] = []

    // Store observations
    state.observations = [...state.observations, ...data.entry]

    // Date filter
    if (state.date !== '') {
      state.observations = state.observations.filter(function (record) {
        var date = new Date(record.resource.meta.lastUpdated)
        return date.getFullYear() === parseInt(state.date.split('-')[0]) &&
               (date.getMonth() + 1) === parseInt(state.date.split('-')[1]) &&
               date.getDate() === parseInt(state.date.split('-')[2])
      })
    }

    // Sort by lastUpdated inside meta field
    state.observations.sort(function (a, b) {
      return new Date(a.resource.meta.lastUpdated).getTime() <= new Date(b.resource.meta.lastUpdated).getTime()
    })

    // Select Weight from observations
    var tempArray = []
    state.observations.forEach(function (element) {
      if (['resource', 'code', 'text'].reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, element) === 'Weight') {
        tempArray.push({date: new Date(element.resource.effectiveDateTime), value: element.resource.valueQuantity.value})
      }
    })
    tempArray.sort(function (a, b) {
      return a.date.getTime() > b.date.getTime()
    })
    tempArray.forEach(function (element) {
      state.dataGraph[0].push(element.value)
      state.dataGraph[1].push(element.date.toLocaleString())
    })

    // Load more content or reset
    if (data.link[1] && data.link[1].relation === 'next') {
      state.patientDetailsURL = data.link[1].url
      state.loadingObservations = true
    } else {
      state.loadingObservations = false
    }
  },
  setList (state, data) {
    state.patients = [...state.patients, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.nextUrl = data.link[1].url
      state.loadingPatients = true
    } else {
      state.nextUrl = constPaths.PATIENT_URL
      state.loadingPatients = false
    }
  },
  filterByDate (state, filteredDate) {
    state.date = filteredDate
    state.observations = state.observations.filter(function (record) {
      var date = new Date(record.resource.meta.lastUpdated)
      return date.getFullYear() === parseInt(filteredDate.split('-')[0]) &&
             (date.getMonth() + 1) === parseInt(filteredDate.split('-')[1]) &&
             date.getDate() === parseInt(filteredDate.split('-')[2])
    })
  },
  clearDateFilter (state) {
    state.observations = []
    state.loadingObservations = true
    state.date = ''
    state.patientDetailsURL = constPaths.PATIENT_URL + state.patientID + '/$everything?_sort_by=date'
  },
  clear (state) {
    state.patients = []
    state.dataGraph[0] = []
    state.dataGraph[1] = []
    state.date = ''
    state.nextUrl = constPaths.PATIENT_URL
    state.loadingPatients = true
  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
